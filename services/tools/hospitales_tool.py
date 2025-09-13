import logging
import time
import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import googlemaps
from rapidfuzz import fuzz, process

from .base import ModernBaseTool, Decision, ToolResult, MatchDetails
from config.constants import information_emojis
from config.settings import GOOGLE_MAPS_API_KEY

logger = logging.getLogger(__name__)


@dataclass
class HospitalData:
    """Estructura de datos para información de hospitales"""
    name: str
    address: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    specialties: List[str] = None
    udh_code: Optional[str] = None
    
    def __post_init__(self):
        if self.specialties is None:
            self.specialties = []


class HospitalesTool(ModernBaseTool):
    """
    Herramienta moderna para consultas sobre hospitales y UDH de la Facultad de Medicina UBA.
    
    Características:
    - Extracción real de datos del mapa de Google Maps My Maps
    - Fuzzy matching avanzado para nombres de hospitales
    - Caché inteligente para optimizar performance
    - Búsqueda por proximidad geográfica
    - Integración con Google Maps API
    """
    
    def __init__(self, google_maps_api_key: Optional[str] = None):
        # Usar API key de configuración si no se proporciona una
        if google_maps_api_key is None:
            google_maps_api_key = GOOGLE_MAPS_API_KEY
        
        # Configuración específica para hospitales
        hospitales_config = {
            'thresholds': {'accept': 0.3},
            'triggers': {
                'keywords': [
                    'hospital', 'hospitales', 'udh', 'unidad docente hospitalaria',
                    'donde queda', 'dónde queda', 'ubicacion', 'ubicación',
                    'direccion', 'dirección', 'localizacion', 'localización',
                    'mapa', 'como llegar', 'cómo llegar', 'donde esta', 'dónde está',
                    'hospital durand', 'hospital clínicas', 'hospital italiano',
                    'hospital alemán', 'hospital británico', 'hospital francés',
                    'hospital santojanni', 'hospital pirovano', 'hospital ramos mejia',
                    'hospital penna', 'hospital muñiz', 'hospital tornu',
                    'hospital zubizarreta', 'hospital alvarez', 'hospital paroissien',
                    'hospital pacheco', 'hospital santamarina', 'hospital bocalandro',
                    'hospital pinto', 'hospital lagleyze', 'hospital ortiz basualdo',
                    'hospital naval', 'hospital militar', 'hospital policia',
                    'hospital de niños', 'hospital pediátrico', 'maternidad',
                    'maternidad suizo argentina', 'maternidad sarda', 'maternidad otamendi',
                    'sanatorio', 'clinica', 'clínica', 'centro medico', 'centro médico'
                ]
            },
            'fuzzy_matching': {
                'enabled': True,
                'threshold': 0.7,
                'weights': {
                    'ratio': 0.3,
                    'partial': 0.25,
                    'token_sort': 0.25,
                    'token_set': 0.2
                }
            },
            'caching': {
                'enabled': True,
                'ttl_minutes': 60  # Cache por 1 hora para datos de hospitales
            },
            'google_maps': {
                'api_key': google_maps_api_key,
                'enabled': bool(google_maps_api_key)
            }
        }
        
        super().__init__("hospitales", 75, hospitales_config)
        
        # URLs del mapa de hospitales
        self.mapa_url = "https://www.google.com/maps/d/u/2/viewer?mid=10mHF_-M-zk2Qfps1dOcThxXUyoKKEPg&ll=-34.62269329999999,-58.94876090000002&z=18"
        self.tiny_url = "https://tinyurl.com/MAPA-UDH-CECIM"
        
        # Inicializar cliente de Google Maps si hay API key
        self.gmaps = None
        if self.config.get('google_maps', {}).get('enabled', False):
            try:
                self.gmaps = googlemaps.Client(key=google_maps_api_key)
                logger.info("Cliente de Google Maps inicializado correctamente")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Google Maps client: {e}")
        
        # Lista de hospitales conocidos para mejor detección
        self.hospitales_conocidos = [
            'durand', 'clínicas', 'italiano', 'alemán', 'británico', 'francés',
            'santojanni', 'pirovano', 'ramos mejia', 'penna', 'muñiz', 'tornu',
            'zubizarreta', 'alvarez', 'paroissien', 'pacheco', 'santamarina',
            'bocalandro', 'pinto', 'lagleyze', 'ortiz basualdo', 'naval',
            'militar', 'policia', 'niños', 'pediátrico', 'maternidad',
            'suizo argentina', 'sarda', 'otamendi'
        ]
        
        # Cache para datos de hospitales
        self._hospitales_cache = None
        self._last_hospitales_fetch = 0

    def _fetch_hospitales_data(self) -> List[HospitalData]:
        """
        Extrae datos de hospitales usando EXCLUSIVAMENTE Google Maps API.
        NO usa web scraping ni datos hardcodeados.
        """
        cache_key = "hospitales_data"
        
        def fetch_data():
            logger.info("Extrayendo datos de hospitales usando Google Maps API...")
            
            if not self.gmaps:
                logger.error("Google Maps API no está configurada")
                return []
            
            # Extraer hospitales usando Google Maps API
            hospitales = self._extract_hospitals_from_google_maps()
            
            if not hospitales:
                logger.error("No se pudieron extraer datos de hospitales desde Google Maps API")
                return []
            
            logger.info(f"Extraídos {len(hospitales)} hospitales desde Google Maps API")
            return hospitales
        
        return self._get_cached_data(cache_key, fetch_data)

    def _extract_hospitals_from_google_maps(self) -> List[HospitalData]:
        """
        Extrae hospitales usando EXCLUSIVAMENTE Google Maps API.
        Busca hospitales en múltiples ubicaciones de Buenos Aires para cubrir toda el área.
        """
        try:
            hospitales = []
            
            # Ubicaciones estratégicas en Buenos Aires para buscar hospitales
            locations = [
                (-34.6037, -58.3816),  # Centro de Buenos Aires
                (-34.6118, -58.3960),  # Palermo
                (-34.5895, -58.3974),  # Recoleta
                (-34.6097, -58.3731),  # San Telmo
                (-34.6345, -58.3731),  # La Boca
                (-34.5955, -58.4019),  # Retiro
                (-34.6200, -58.4200),  # Caballito
                (-34.6400, -58.3800),  # Barracas
                (-34.6275, -58.0079),  # Coordenadas del mapa original
            ]
            
            # Radio de búsqueda en metros
            radius = 3000  # 3km por ubicación para evitar duplicados
            
            for location in locations:
                try:
                    # Buscar hospitales en esta ubicación
                    places_result = self.gmaps.places_nearby(
                        location=location,
                        radius=radius,
                        type='hospital'
                    )
                    
                    # Procesar resultados
                    for place in places_result.get('results', []):
                        hospital = self._create_hospital_from_place(place)
                        if hospital and not self._is_duplicate_hospital(hospital, hospitales):
                            hospitales.append(hospital)
                    
                    # Pequeña pausa para evitar límites de rate
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error buscando hospitales en {location}: {e}")
                    continue
            
            # Eliminar duplicados finales
            hospitales = self._remove_duplicate_hospitals(hospitales)
            
            logger.info(f"Extraídos {len(hospitales)} hospitales únicos desde Google Maps API")
            return hospitales
            
        except Exception as e:
            logger.error(f"Error extrayendo hospitales desde Google Maps API: {e}")
            return []
    
    def _create_hospital_from_place(self, place: Dict[str, Any]) -> Optional[HospitalData]:
        """
        Crea un objeto HospitalData desde un resultado de Google Places API.
        """
        try:
            name = place.get('name', '').strip()
            if not name:
                return None
            
            # Filtrar solo hospitales, sanatorios y clínicas
            name_lower = name.lower()
            if not any(keyword in name_lower for keyword in ['hospital', 'sanatorio', 'clínica', 'clinica', 'centro médico', 'centro medico']):
                return None
            
            hospital = HospitalData(
                name=name,
                address=place.get('vicinity', ''),
                coordinates=(
                    place['geometry']['location']['lat'],
                    place['geometry']['location']['lng']
                )
            )
            
            # Obtener información adicional si está disponible
            if 'place_id' in place:
                hospital.udh_code = f"UDH-{place['place_id'][:8]}"
            
            return hospital
            
        except Exception as e:
            logger.warning(f"Error creando hospital desde place: {e}")
            return None
    
    def _is_duplicate_hospital(self, new_hospital: HospitalData, existing_hospitals: List[HospitalData]) -> bool:
        """
        Verifica si un hospital ya existe en la lista basándose en nombre y coordenadas.
        """
        for existing in existing_hospitals:
            # Comparar nombres (fuzzy matching)
            name_similarity = fuzz.ratio(new_hospital.name.lower(), existing.name.lower())
            if name_similarity > 85:  # 85% de similitud
                return True
            
            # Comparar coordenadas (muy cerca)
            if (new_hospital.coordinates and existing.coordinates and 
                self._calculate_distance(new_hospital.coordinates, existing.coordinates) < 0.1):  # 100 metros
                return True
        
        return False
    
    
    def _remove_duplicate_hospitals(self, hospitales: List[HospitalData]) -> List[HospitalData]:
        """Elimina hospitales duplicados basándose en el nombre."""
        seen_names = set()
        unique_hospitals = []
        
        for hospital in hospitales:
            name_lower = hospital.name.lower().strip()
            if name_lower not in seen_names and len(name_lower) > 5:
                seen_names.add(name_lower)
                unique_hospitals.append(hospital)
        
        return unique_hospitals



    def _rule_score(self, query: str) -> float:
        """
        Calcula el score basado en keywords y patrones específicos con fuzzy matching.
        """
        query_norm = self._normalize_query(query)
        keywords = self.config.get('triggers', {}).get('keywords', [])
        
        # Score base por keywords con fuzzy matching
        score, match_details = self._advanced_keyword_matching(query_norm, keywords)
        
        # Boost si menciona hospitales específicos
        hospital_hits = sum(1 for hospital in self.hospitales_conocidos if hospital in query_norm)
        hospital_boost = min(0.3, 0.1 * hospital_hits)
        
        # Boost por patrones de ubicación
        ubicacion_patterns = ['donde queda', 'dónde queda', 'ubicacion', 'ubicación', 'direccion', 'dirección']
        ubicacion_boost = 0.2 if any(pattern in query_norm for pattern in ubicacion_patterns) else 0.0
        
        # Boost por patrones de mapa/navegación
        mapa_patterns = ['mapa', 'como llegar', 'cómo llegar', 'localizacion', 'localización']
        mapa_boost = 0.2 if any(pattern in query_norm for pattern in mapa_patterns) else 0.0
        
        total_score = score + hospital_boost + ubicacion_boost + mapa_boost
        return min(1.0, total_score)

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        """
        Determina si la tool puede manejar la consulta con contexto avanzado.
        """
        score = self._rule_score(query)
        
        # Boost si el contexto previo fue sobre hospitales
        last_qt = (context or {}).get('last_query_type', '')
        if isinstance(last_qt, str) and 'hospital' in last_qt.lower():
            score = max(score, 0.8)
        
        # Boost por contexto geográfico
        if 'buenos aires' in query.lower() or 'caba' in query.lower():
            score = max(score, score + 0.1)
        
        return Decision(
            score=score, 
            params={}, 
            reasons=["hospitales_rule_score", "context_boost"]
        )

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Ejecuta la funcionalidad principal con extracción de datos reales.
        """
        try:
            query_norm = self._normalize_query(query)
            
            # Obtener datos de hospitales
            hospitales = self._fetch_hospitales_data()
            
            # Detectar si menciona un hospital específico
            hospital_mencionado = self._find_mentioned_hospital(query_norm, hospitales)
            
            # Generar respuesta personalizada
            if hospital_mencionado:
                response = self._generate_specific_hospital_response(hospital_mencionado, hospitales)
            else:
                response = self._generate_general_hospital_response(hospitales)
            
            return ToolResult(
                response=response,
                sources=["Mapa de Hospitales UDH-CECIM", "Google Maps API"] if self.gmaps else ["Mapa de Hospitales UDH-CECIM"],
                metadata={
                    'hospital_mentioned': hospital_mencionado.name if hospital_mencionado else None,
                    'total_hospitals': len(hospitales),
                    'data_source': 'extracted' if hospitales else 'static'
                }
            )
            
        except Exception as e:
            logger.error(f"Error ejecutando HospitalesTool: {e}")
            return ToolResult(
                response="❌ Error al procesar la consulta sobre hospitales. Por favor, intenta nuevamente.",
                sources=["Sistema de Hospitales"],
                metadata={'error': str(e)}
            )

    def _find_mentioned_hospital(self, query: str, hospitales: List[HospitalData]) -> Optional[HospitalData]:
        """
        Encuentra el hospital mencionado en la consulta usando fuzzy matching.
        """
        if not hospitales:
            return None
        
        # Crear lista de nombres de hospitales para matching
        hospital_names = [h.name.lower() for h in hospitales]
        
        # Buscar coincidencia exacta primero
        for hospital in hospitales:
            if hospital.name.lower() in query:
                return hospital
        
        # Buscar con fuzzy matching
        best_match = process.extractOne(query, hospital_names, scorer=fuzz.ratio)
        if best_match and best_match[1] >= 60:  # Threshold de 60%
            matched_name = best_match[0]
            for hospital in hospitales:
                if hospital.name.lower() == matched_name:
                    return hospital
        
        return None

    def _generate_specific_hospital_response(self, hospital: HospitalData, all_hospitals: List[HospitalData]) -> str:
        """
        Genera respuesta para un hospital específico con información detallada.
        """
        emoji = information_emojis[0] if information_emojis else "🏥"
        
        response_parts = [
            f"{emoji} **{hospital.name}**",
            "",
            f"📍 **Ubicación:** {hospital.address or 'Dirección no disponible'}",
        ]
        
        if hospital.coordinates:
            lat, lng = hospital.coordinates
            response_parts.append(f"🗺️ **Coordenadas:** {lat:.6f}, {lng:.6f}")
        
        if hospital.udh_code:
            response_parts.append(f"🏛️ **Código UDH:** {hospital.udh_code}")
        
        response_parts.extend([
            "",
            f"🗺️ **Mapa interactivo:** {self.tiny_url}",
            "",
            f"En este mapa encontrarás la ubicación exacta del {hospital.name} junto con "
            f"todos los demás hospitales y Unidades Docentes Hospitalarias (UDH) "
            f"donde se realizan las prácticas de la carrera de Medicina."
        ])
        
        return "\n".join(response_parts)

    def _generate_general_hospital_response(self, hospitales: List[HospitalData]) -> str:
        """
        Genera respuesta general para consultas sobre hospitales con estadísticas.
        """
        emoji = information_emojis[0] if information_emojis else "🏥"
        
        response_parts = [
            f"{emoji} **Hospitales y UDH de la Facultad de Medicina UBA**",
            "",
            f"📊 **Total de hospitales disponibles:** {len(hospitales)}",
            "",
            f"🗺️ **Mapa interactivo:** {self.tiny_url}",
            "",
            "**Algunos hospitales principales:**"
        ]
        
        # Mostrar algunos hospitales principales
        main_hospitals = hospitales[:8]  # Primeros 8 hospitales
        for hospital in main_hospitals:
            response_parts.append(f"• {hospital.name}")
        
        if len(hospitales) > 8:
            response_parts.append(f"• Y {len(hospitales) - 8} hospitales más...")
        
        response_parts.extend([
            "",
            "En este mapa encontrarás la ubicación de todos los hospitales donde se "
            "realizan las prácticas de la carrera de Medicina, incluyendo sus direcciones "
            "exactas y códigos de UDH correspondientes.",
            "",
            "¡El mapa te permitirá ubicar fácilmente cualquier hospital o UDH!"
        ])
        
        return "\n".join(response_parts)

    def get_hospital_by_name(self, name: str) -> Optional[HospitalData]:
        """
        Método público para obtener información de un hospital específico.
        """
        hospitales = self._fetch_hospitales_data()
        return self._find_mentioned_hospital(name.lower(), hospitales)

    def search_hospitals_nearby(self, coordinates: Tuple[float, float], radius_km: float = 10.0) -> List[HospitalData]:
        """
        Busca hospitales cerca de unas coordenadas específicas.
        """
        hospitales = self._fetch_hospitales_data()
        nearby_hospitals = []
        
        for hospital in hospitales:
            if hospital.coordinates:
                distance = self._calculate_distance(coordinates, hospital.coordinates)
                if distance <= radius_km:
                    hospital.distance_km = distance  # Agregar distancia como atributo
                    nearby_hospitals.append(hospital)
        
        # Ordenar por distancia
        nearby_hospitals.sort(key=lambda h: getattr(h, 'distance_km', float('inf')))
        return nearby_hospitals

    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calcula la distancia entre dos coordenadas usando la fórmula de Haversine.
        """
        from math import radians, cos, sin, asin, sqrt
        
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radio de la Tierra en kilómetros
        r = 6371
        return c * r
    
    
    def search_hospitals_by_neighborhood(self, neighborhood: str) -> List[HospitalData]:
        """
        Busca hospitales por barrio usando Google Maps API.
        """
        if not self.gmaps:
            logger.warning("Google Maps API no está configurada")
            return []
        
        try:
            # Buscar el barrio en Buenos Aires
            geocode_result = self.gmaps.geocode(f"{neighborhood}, Buenos Aires, Argentina")
            
            if not geocode_result:
                logger.warning(f"No se encontró el barrio: {neighborhood}")
                return []
            
            location = geocode_result[0]['geometry']['location']
            
            # Buscar hospitales cerca del barrio
            places_result = self.gmaps.places_nearby(
                location=(location['lat'], location['lng']),
                radius=5000,  # 5km de radio
                type='hospital'
            )
            
            hospitales = []
            for place in places_result.get('results', []):
                hospital = self._create_hospital_from_place(place)
                if hospital:
                    hospitales.append(hospital)
            
            logger.info(f"Encontrados {len(hospitales)} hospitales en {neighborhood}")
            return hospitales
            
        except Exception as e:
            logger.error(f"Error buscando hospitales por barrio: {e}")
            return []