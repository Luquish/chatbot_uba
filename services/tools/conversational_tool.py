import logging
from typing import Any, Dict, List, Optional

from .base import ConversationalBaseTool, Decision, ToolResult
from handlers.intent_handler import generate_conversational_response, get_intent_by_keywords, normalize_intent_examples
from config.constants import INTENT_EXAMPLES
from models.openai_model import OpenAIModel


logger = logging.getLogger(__name__)


class ConversationalTool(ConversationalBaseTool):
    """
    Tool para manejar conversaciones generales y saludos.
    Hereda de ConversationalBaseTool para funcionalidades modernas.
    """
    
    name = "conversational"
    priority = 85

    def __init__(self, model: Optional[OpenAIModel] = None):
        # Configuración específica para conversacional
        default_config = {
            'thresholds': {
                'accept': 0.5
            },
            'triggers': {
                'keywords': [
                    'hola', 'buenas', 'buenos dias', 'buen dia', 'que tal',
                    'como estas', 'gracias', 'quien eres', 'como te llamas', 
                    'que puedes hacer', 'que podes hacer', 'help', 'ayuda',
                    'buenas tardes', 'buenas noches', 'saludos', 'hi', 'hello'
                ]
            },
            'context_awareness': {
                'enabled': True,
                'context_boost': 0.3
            },
            'intent_detection': {
                'enabled': True,
                'use_llm': True
            }
        }
        
        # Usar constructor de ConversationalBaseTool
        super().__init__(self.name, self.priority, model, default_config)
        
        # Configuración específica para intents
        self._normalized_examples = normalize_intent_examples(INTENT_EXAMPLES)

    def configure(self, config: Dict[str, Any]) -> None:
        """Configurar la tool con parámetros específicos."""
        if not config:
            return
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        """
        Calcula score específico para conversacional.
        Usa la lógica base y añade boosts específicos.
        """
        # Usar score base de la clase padre
        base_score = super()._rule_score(query)
        
        # Boost adicional por longitud (consultas cortas son más conversacionales)
        query_words = len(query.split())
        if query_words <= 2:
            base_score = max(base_score, 0.8)
        elif query_words <= 4:
            base_score = max(base_score, 0.6)
        
        # Boost por palabras de saludo específicas
        greeting_words = ['hola', 'hi', 'hello', 'buenas', 'saludos']
        query_lower = query.lower()
        greeting_boost = sum(0.1 for word in greeting_words if word in query_lower)
        
        return min(1.0, base_score + greeting_boost)

    def _detect_intent(self, query: str, context: Dict[str, Any]) -> tuple[str, float]:
        """
        Detecta la intención de la consulta.
        Combina detección por keywords con lógica conversacional.
        """
        # Usar detección base
        intent, confidence = super()._detect_intent(query, context)
        
        # Mejorar con keywords específicas
        kw = get_intent_by_keywords(query, self._normalized_examples)
        if kw:
            keyword_intent = kw[0]
            keyword_confidence = kw[1] if len(kw) > 1 else 0.8
            
            # Usar el intent con mayor confianza
            if keyword_confidence > confidence:
                intent = keyword_intent
                confidence = keyword_confidence
        
        return intent, confidence

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """
        Ejecuta la respuesta conversacional.
        """
        try:
            # Detectar intent usando el método mejorado
            intent, confidence = self._detect_intent(query, context)
            
            # Obtener nombre de usuario del contexto
            user_name = context.get('user_name') if isinstance(context, dict) else None
            
            # Generar respuesta usando el handler existente
            if self.model:
                text = generate_conversational_response(
                    self.model, query, intent, user_name=user_name
                )
            else:
                # Fallback si no hay modelo
                text = self._generate_fallback_response(query, intent, user_name)
            
            # Metadata con información del intent detectado
            metadata = {
                'intent': intent,
                'confidence': confidence,
                'query_length': len(query.split()),
                'user_name': user_name
            }
            
            return ToolResult(
                response=text, 
                sources=["Conversational LLM"], 
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error en ConversationalTool: {str(e)}")
            return ToolResult(
                response="Disculpa, hubo un problema procesando tu mensaje. ¿Puedes intentar de nuevo?",
                sources=["Error Handler"],
                metadata={'error': str(e)}
            )

    def _generate_fallback_response(self, query: str, intent: str, user_name: Optional[str] = None) -> str:
        """
        Genera respuesta de fallback cuando no hay modelo disponible.
        """
        greeting = f"Hola {user_name}!" if user_name else "¡Hola!"
        
        if intent == "greeting":
            return f"{greeting} ¿En qué puedo ayudarte hoy?"
        elif intent == "gratitude":
            return "¡De nada! Estoy aquí para ayudarte cuando lo necesites."
        elif intent == "help":
            return f"{greeting} Puedo ayudarte con información sobre horarios, contactos de hospitales, cursos y más. ¿Qué necesitas saber?"
        else:
            return f"{greeting} ¿En qué puedo asistirte?"


