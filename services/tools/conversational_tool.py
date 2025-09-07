import logging
from typing import Any, Dict, List

from .base import BaseTool, Decision, ToolResult
from handlers.intent_handler import generate_conversational_response, get_intent_by_keywords, normalize_intent_examples
from config.constants import INTENT_EXAMPLES
from models.openai_model import OpenAIModel


logger = logging.getLogger(__name__)


class ConversationalTool:
    name = "conversational"
    priority = 85

    def __init__(self, model: OpenAIModel):
        self.model = model
        self.config: Dict[str, Any] = {
            'thresholds': {
                'accept': 0.5
            },
            'triggers': {
                # Triggers mínimos; la detección real usa LLM y keywords ligeras
                'keywords': [
                    'hola', 'buenas', 'buenos dias', 'buen dia', 'que tal',
                    'como estas', 'gracias', 'quien eres', 'como te llamas', 'que puedes hacer', 'que podes hacer'
                ]
            }
        }
        self._normalized_examples = normalize_intent_examples(INTENT_EXAMPLES)

    def configure(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        query_l = query.lower()
        keywords: List[str] = self.config.get('triggers', {}).get('keywords', [])
        hits = sum(1 for k in keywords if k in query_l)
        return min(1.0, 0.2 * hits) if hits else 0.0

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        # Score por triggers + boost si la consulta es corta (típico de saludos)
        score = self._rule_score(query)
        if len(query.split()) <= 4:
            score = max(score, 0.6)
        return Decision(score=score, params={}, reasons=["conversational_rule_score"])

    def accepts(self, score: float) -> bool:
        accept = float(self.config.get('thresholds', {}).get('accept', 0.5))
        return score >= accept

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        # Intent ligero por keyword; si no hay, dejar al LLM decidir en generate_conversational_response
        kw = get_intent_by_keywords(query, self._normalized_examples)
        intent = kw[0] if kw else 'desconocido'
        user_name = context.get('user_name') if isinstance(context, dict) else None
        text = generate_conversational_response(self.model, query, intent, user_name=user_name)
        return ToolResult(response=text, sources=["Conversational LLM"], metadata={})


