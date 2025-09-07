import logging
from typing import Any, Dict, List

from .base import BaseTool, Decision, ToolResult
from handlers.faqs_handler import handle_faq_query


logger = logging.getLogger(__name__)


class FaqTool:
    name = "faq"
    priority = 80

    def __init__(self):
        self.config: Dict[str, Any] = {
            'thresholds': {
                'accept': 0.7
            },
            'triggers': {
                'keywords': []
            }
        }

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
        score = self._rule_score(query)
        return Decision(score=score, params={}, reasons=["faq_rule_score"])

    def accepts(self, score: float) -> bool:
        accept = float(self.config.get('thresholds', {}).get('accept', 0.7))
        return score >= accept

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        text = handle_faq_query(query)
        if not text:
            return ToolResult(response="", sources=[], metadata={})
        return ToolResult(response=text, sources=["Preguntas Frecuentes"], metadata={})


