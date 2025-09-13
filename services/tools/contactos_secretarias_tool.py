"""
Tool para consultar contactos de secretarías de hospitales.
Proporciona información de mails de contacto para diferentes hospitales y departamentos.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from services.tools.base import SheetsBaseTool, MatchDetails


class ContactosSecretariasTool(SheetsBaseTool):
    """
    Tool para consultar contactos de secretarías de hospitales.
    
    Busca en una hoja de Google Sheets que contiene:
    - Columna A: Nombres de hospitales/departamentos
    - Columna B: Mails de contacto
    """
    
    name = "sheets.contactos_secretarias"
    priority = 70

    def __init__(self, sheets_service: Optional[Any]):
        # Configuración específica para contactos de secretarías
        default_config = {
            'thresholds': {'accept': 0.5},
            'triggers': {
                'keywords': [
                    # Keywords básicas
                    'contacto', 'mail', 'email', 'correo', 'secretaria', 'secretaría',
                    'hospital', 'hospitales', 'udh', 'unidad docente hospitalaria',
                    # Nombres de hospitales específicos
                    'aeronautico', 'alvarez', 'alvear', 'argerich', 'avellaneda', 'borda',
                    'churruca', 'clinicas', 'clínicas', 'catedra', 'cátedra',
                    # Departamentos
                    'medicina', 'cirugia', 'cirugía', 'dermatologia', 'dermatología',
                    'primera', 'segunda', 'tercera', 'cuarta', 'quinta', 'sexta', 'septima',
                    '1ra', '2da', '3ra', '4ta', '5ta', '6ta', '7ma',
                    # Términos de búsqueda
                    'donde', 'dónde', 'ubicacion', 'ubicación', 'direccion', 'dirección'
                ]
            },
            'spreadsheet_id': "19t4GasPKs2_Th-2YofT62KgPtfkJDlhLr7fR8Y2iJ4I",
            'sheet_name': 'Hoja 1',
            'ranges': {'default': 'A:B'},  # Solo Hospital y Mail
            'caching': {
                'enabled': True,
                'ttl_minutes': 60  # Cache más largo para contactos
            },
            'fuzzy_matching': {
                'enabled': True,
                'threshold': 0.6,
                'weights': {
                    'ratio': 0.4,
                    'partial': 0.3,
                    'token_sort': 0.2,
                    'token_set': 0.1
                }
            }
        }
        
        # Usar constructor de SheetsBaseTool
        super().__init__(self.name, self.priority, sheets_service, default_config)
        
        # Alias para mejorar matching de hospitales
        self.hospital_aliases = {
            'clinicas': 'clínicas',
            'catedra': 'cátedra',
            'cirugia': 'cirugía',
            'dermatologia': 'dermatología',
            'medicina': 'medicina',
            'primera': '1ra',
            'segunda': '2da',
            'tercera': '3ra',
            'cuarta': '4ta',
            'quinta': '5ta',
            'sexta': '6ta',
            'septima': '7ma'
        }
        
        # Stop words para limpiar consultas
        self.stop_words = {
            'de', 'del', 'la', 'el', 'donde', 'dónde', 'está', 'esta', 'queda',
            'contacto', 'mail', 'email', 'correo', 'secretaria', 'secretaría',
            'hospital', 'udh', 'unidad', 'docente', 'hospitalaria'
        }

    def configure(self, config: Dict[str, Any]) -> None:
        """Configurar la tool con parámetros específicos."""
        if not config:
            return
        
        # Manejar spreadsheet_id desde variables de entorno si es necesario
        sid = config.get('spreadsheet_id')
        if isinstance(sid, str) and sid.startswith('ENV:'):
            env_key = sid.split(':', 1)[1]
            config['spreadsheet_id'] = os.getenv(env_key)
        
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        """
        Calcular score basado en reglas específicas para contactos de hospitales.
        """
        # Usar score base de la clase padre
        base_score = super()._rule_score(query)
        
        # Boost para términos específicos de hospitales
        hospital_terms = [
            'aeronautico', 'alvarez', 'alvear', 'argerich', 'avellaneda', 'borda',
            'churruca', 'clinicas', 'clínicas', 'catedra', 'cátedra'
        ]
        
        query_lower = query.lower()
        hospital_boost = 0.0
        
        for term in hospital_terms:
            if term in query_lower:
                hospital_boost += 0.1
        
        # Boost para términos de contacto
        contact_terms = ['contacto', 'mail', 'email', 'correo']
        contact_boost = 0.0
        
        for term in contact_terms:
            if term in query_lower:
                contact_boost += 0.05
        
        return min(1.0, base_score + hospital_boost + contact_boost)

    def execute(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Ejecutar búsqueda de contactos de hospitales.
        
        Args:
            query: Consulta del usuario
            context: Contexto adicional (opcional)
            
        Returns:
            Respuesta formateada con contactos encontrados
        """
        try:
            # Obtener datos de la hoja
            rows = self._fetch_sheet_data()
            if not rows:
                return "❌ No se pudieron obtener los datos de contactos de hospitales."
            
            # Procesar filas (saltar encabezados si los hay)
            processed_rows = self._process_sheet_rows(rows)
            if len(processed_rows) < 2:
                return "❌ No hay datos suficientes en la hoja de contactos."
            
            # Normalizar consulta
            normalized_query = self._normalize_query(query)
            
            # Buscar coincidencias
            matches = []
            for i, row in enumerate(processed_rows[1:], start=1):  # Saltar encabezado
                if len(row) >= 2:
                    hospital_name = str(row[0]).strip()
                    mail = str(row[1]).strip()
                    
                    if hospital_name and mail:
                        # Calcular score de coincidencia
                        score, match_details = self._calculate_fuzzy_score(
                            normalized_query, hospital_name
                        )
                        
                        if score >= self.config['fuzzy_matching']['threshold']:
                            matches.append((i, score, match_details, hospital_name, mail))
            
            # Ordenar por score descendente
            matches.sort(key=lambda x: x[1], reverse=True)
            
            if not matches:
                return f"❌ No se encontraron contactos para '{query}'. Intenta con el nombre del hospital o departamento."
            
            # Formatear respuesta
            response_parts = ["🏥 **CONTACTOS DE HOSPITALES**\n"]
            
            # Mostrar hasta 5 resultados
            for i, (row_idx, score, match_details, hospital, mail) in enumerate(matches[:5]):
                response_parts.append(f"📍 **{hospital}**")
                response_parts.append(f"📧 {mail}")
                response_parts.append("")  # Línea en blanco
            
            if len(matches) > 5:
                response_parts.append(f"... y {len(matches) - 5} contactos más.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            return f"❌ Error al buscar contactos: {str(e)}"

    def _normalize_query(self, query: str) -> str:
        """
        Normalizar consulta para mejor matching.
        """
        # Usar normalización base
        normalized = super()._normalize_query(query)
        
        # Aplicar alias de hospitales
        for alias, replacement in self.hospital_aliases.items():
            normalized = normalized.replace(alias, replacement)
        
        return normalized
