import logging
from typing import Any, Dict, List, Optional

from .base import ModernBaseTool, Decision, ToolResult
from handlers.faqs_handler import handle_faq_query


logger = logging.getLogger(__name__)


class FaqTool(ModernBaseTool):
    """
    Tool para manejar preguntas frecuentes (FAQ).
    Hereda de ModernBaseTool para funcionalidades modernas de matching y cache.
    """
    
    name = "faq"
    priority = 80

    def __init__(self):
        # Configuración específica para FAQ
        default_config = {
            'thresholds': {
                'accept': 0.7
            },
            'triggers': {
                'keywords': [
                    # Keywords para tramites y documentación
                    'cómo tramitar', 'como tramitar', 'tramitar', 'tramite', 'trámite',
                    'requisitos', 'documentación', 'documentacion', 'documentos',
                    'dónde', 'donde', 'ubicacion', 'ubicación', 'direccion', 'dirección',
                    'como obtener', 'cómo obtener', 'obtener', 'conseguir',
                    'inscripcion', 'inscripción', 'inscribirse', 'inscribir',
                    'certificado', 'certificados', 'constancia', 'constancias',
                    'legajo', 'legajos', 'expediente', 'expedientes',
                    'pago', 'pagos', 'arancel', 'aranceles', 'costo', 'costos',
                    'fecha', 'fechas', 'plazo', 'plazos', 'vencimiento', 'vencimientos',
                    'horario', 'horarios', 'atencion', 'atención', 'atender',
                    'turno', 'turnos', 'cita', 'citas', 'agendar', 'agendamiento',
                    'informacion', 'información', 'info', 'consulta', 'consultas',
                    'pregunta', 'preguntas', 'duda', 'dudas', 'ayuda',
                    'proceso', 'procesos', 'pasos', 'instrucciones', 'guia', 'guía'
                ]
            },
            'caching': {
                'enabled': True,
                'ttl_minutes': 120  # Cache más largo para FAQ (2 horas)
            },
            'fuzzy_matching': {
                'enabled': True,
                'threshold': 0.6,
                'weights': {
                    'ratio': 0.3,
                    'partial': 0.3,
                    'token_sort': 0.2,
                    'token_set': 0.2
                }
            },
            'faq_specific': {
                'boost_question_words': True,
                'boost_process_words': True,
                'min_query_length': 3
            }
        }
        
        # Usar constructor de ModernBaseTool
        super().__init__(self.name, self.priority, default_config)

    def configure(self, config: Dict[str, Any]) -> None:
        """Configurar la tool con parámetros específicos."""
        if not config:
            return
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        """
        Calcula score específico para FAQ.
        Usa la lógica base y añade boosts específicos para preguntas frecuentes.
        """
        # Usar score base de la clase padre
        base_score = super()._rule_score(query)
        
        # Boost por palabras de pregunta
        if self.config.get('faq_specific', {}).get('boost_question_words', True):
            question_words = [
                'como', 'cómo', 'que', 'qué', 'donde', 'dónde', 'cuando', 'cuándo',
                'por que', 'por qué', 'para que', 'para qué', 'cual', 'cuál',
                'tramitar', 'obtener', 'conseguir', 'inscribir', 'certificar'
            ]
            query_lower = query.lower()
            question_boost = sum(0.05 for word in question_words if word in query_lower)
            base_score = min(1.0, base_score + question_boost)
        
        # Boost por palabras de proceso
        if self.config.get('faq_specific', {}).get('boost_process_words', True):
            process_words = [
                'proceso', 'pasos', 'instrucciones', 'guia', 'guía', 'manual',
                'requisitos', 'documentos', 'documentación', 'papeles'
            ]
            query_lower = query.lower()
            process_boost = sum(0.08 for word in process_words if word in query_lower)
            base_score = min(1.0, base_score + process_boost)
        
        # Penalizar consultas muy cortas
        min_length = self.config.get('faq_specific', {}).get('min_query_length', 3)
        if len(query.split()) < min_length:
            base_score *= 0.5
        
        return base_score

    def _normalize_query(self, query: str) -> str:
        """
        Normalización específica para FAQ.
        """
        # Usar normalización base
        normalized = super()._normalize_query(query)
        
        # Normalizaciones específicas para FAQ
        faq_aliases = {
            'como': 'cómo',
            'donde': 'dónde',
            'cuando': 'cuándo',
            'por que': 'por qué',
            'para que': 'para qué',
            'cual': 'cuál',
            'tramite': 'trámite',
            'documentacion': 'documentación',
            'informacion': 'información',
            'atencion': 'atención',
            'inscripcion': 'inscripción',
            'ubicacion': 'ubicación',
            'direccion': 'dirección',
            'guia': 'guía'
        }
        
        for alias, replacement in faq_aliases.items():
            normalized = normalized.replace(alias, replacement)
        
        return normalized

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Ejecuta la búsqueda en FAQ.
        """
        try:
            # Normalizar consulta
            normalized_query = self._normalize_query(query)
            
            # Usar cache si está habilitado
            cache_key = f"faq_{hash(normalized_query)}"
            
            def fetch_faq_response():
                return handle_faq_query(normalized_query)
            
            # Obtener respuesta (con cache si está habilitado)
            if self.config.get('caching', {}).get('enabled', True):
                ttl = self.config.get('caching', {}).get('ttl_minutes', 120)
                text = self._get_cached_data(cache_key, fetch_faq_response, ttl)
            else:
                text = fetch_faq_response()
            
            # Verificar si se encontró respuesta
            if not text or text.strip() == "":
                return ToolResult(
                    response="",
                    sources=[],
                    metadata={
                        'query': normalized_query,
                        'found': False,
                        'cache_used': self.config.get('caching', {}).get('enabled', True)
                    }
                )
            
            # Metadata con información de la consulta
            metadata = {
                'query': normalized_query,
                'found': True,
                'response_length': len(text),
                'cache_used': self.config.get('caching', {}).get('enabled', True),
                'fuzzy_threshold': self.config.get('fuzzy_matching', {}).get('threshold', 0.6)
            }
            
            return ToolResult(
                response=text,
                sources=["Preguntas Frecuentes"],
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error en FaqTool: {str(e)}")
            return ToolResult(
                response="",
                sources=[],
                metadata={
                    'error': str(e),
                    'query': query,
                    'found': False
                }
            )


