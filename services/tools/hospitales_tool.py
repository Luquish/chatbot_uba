import logging
from typing import Any, Dict, List
import random

from .base import BaseTool, Decision, ToolResult
from config.constants import information_emojis

logger = logging.getLogger(__name__)


class HospitalesTool:
    name = "hospitales"
    priority = 75

    def __init__(self):
        self.config: Dict[str, Any] = {
            'thresholds': {
                'accept': 0.4
            },
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
            }
        }
        
        # URL del mapa de hospitales
        self.mapa_url = "https://tinyurl.com/MAPA-UDH-CECIM"
        
        # Lista de hospitales conocidos para mejor detección
        self.hospitales_conocidos = [
            'durand', 'clínicas', 'italiano', 'alemán', 'británico', 'francés',
            'santojanni', 'pirovano', 'ramos mejia', 'penna', 'muñiz', 'tornu',
            'zubizarreta', 'alvarez', 'paroissien', 'pacheco', 'santamarina',
            'bocalandro', 'pinto', 'lagleyze', 'ortiz basualdo', 'naval',
            'militar', 'policia', 'niños', 'pediátrico', 'maternidad',
            'suizo argentina', 'sarda', 'otamendi'
        ]

    def configure(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        """Calcula el score basado en keywords y patrones específicos."""
        query_l = query.lower()
        keywords: List[str] = self.config.get('triggers', {}).get('keywords', [])
        
        # Score base por keywords
        hits = sum(1 for k in keywords if k in query_l)
        base_score = min(1.0, 0.15 * hits) if hits else 0.0
        
        # Boost si menciona hospitales específicos
        hospital_hits = sum(1 for hospital in self.hospitales_conocidos if hospital in query_l)
        hospital_boost = min(0.3, 0.1 * hospital_hits)
        
        # Boost por patrones de ubicación
        ubicacion_patterns = ['donde queda', 'dónde queda', 'ubicacion', 'ubicación', 'direccion', 'dirección']
        ubicacion_boost = 0.2 if any(pattern in query_l for pattern in ubicacion_patterns) else 0.0
        
        # Boost por patrones de mapa/navegación
        mapa_patterns = ['mapa', 'como llegar', 'cómo llegar', 'localizacion', 'localización']
        mapa_boost = 0.2 if any(pattern in query_l for pattern in mapa_patterns) else 0.0
        
        total_score = base_score + hospital_boost + ubicacion_boost + mapa_boost
        return min(1.0, total_score)

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        score = self._rule_score(query)
        
        # Boost si el contexto previo fue sobre hospitales
        last_qt = (context or {}).get('last_query_type', '')
        if isinstance(last_qt, str) and 'hospital' in last_qt.lower():
            score = max(score, 0.8)
        
        return Decision(score=score, params={}, reasons=["hospitales_rule_score"])

    def accepts(self, score: float) -> bool:
        accept = float(self.config.get('thresholds', {}).get('accept', 0.6))
        return score >= accept

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """Genera respuesta con el mapa de hospitales."""
        query_l = query.lower()
        
        # Detectar si menciona un hospital específico
        hospital_mencionado = None
        for hospital in self.hospitales_conocidos:
            if hospital in query_l:
                hospital_mencionado = hospital.title()
                break
        
        # Generar respuesta personalizada
        if hospital_mencionado:
            response = self._generate_specific_hospital_response(hospital_mencionado)
        else:
            response = self._generate_general_hospital_response()
        
        return ToolResult(
            response=response, 
            sources=["Mapa de Hospitales UDH-CECIM"], 
            metadata={'hospital_mentioned': hospital_mencionado}
        )

    def _generate_specific_hospital_response(self, hospital: str) -> str:
        """Genera respuesta para un hospital específico."""
        emoji = random.choice(information_emojis)
        return (
            f"{emoji} Para encontrar la ubicación del **{hospital}** y otros hospitales "
            f"de la Facultad de Medicina UBA, puedes consultar nuestro mapa interactivo:\n\n"
            f"🗺️ **Mapa de Hospitales y UDH:**\n"
            f"{self.mapa_url}\n\n"
            f"En este mapa encontrarás la ubicación exacta del {hospital} junto con "
            f"todos los demás hospitales y Unidades Docentes Hospitalarias (UDH) "
            f"donde se realizan las prácticas de la carrera de Medicina."
        )

    def _generate_general_hospital_response(self) -> str:
        """Genera respuesta general para consultas sobre hospitales."""
        emoji = random.choice(information_emojis)
        return (
            f"{emoji} Para encontrar la ubicación de los hospitales y Unidades Docentes "
            f"Hospitalarias (UDH) de la Facultad de Medicina UBA, puedes consultar "
            f"nuestro mapa interactivo:\n\n"
            f"🗺️ **Mapa de Hospitales y UDH:**\n"
            f"{self.mapa_url}\n\n"
            f"En este mapa encontrarás la ubicación de todos los hospitales donde se "
            f"realizan las prácticas de la carrera de Medicina, incluyendo:\n"
            f"• Hospital Durand\n"
            f"• Hospital de Clínicas\n"
            f"• Hospital Italiano\n"
            f"• Hospital Alemán\n"
            f"• Y muchos más...\n\n"
            f"¡El mapa te permitirá ubicar fácilmente cualquier hospital o UDH!"
        )
