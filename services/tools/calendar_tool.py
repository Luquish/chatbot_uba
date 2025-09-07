import logging
from typing import Any, Dict

from .base import BaseTool, Decision, ToolResult
from services.calendar_service import CalendarService
from handlers.calendar_handler import get_calendar_events
from config.constants import CALENDAR_INTENT_MAPPING


logger = logging.getLogger(__name__)


class CalendarTool:
    name = "calendar"
    priority = 70

    def __init__(self, calendar_service: CalendarService | None):
        self.service = calendar_service
        self.config: Dict[str, Any] = {
            'thresholds': {
                'accept': 0.65
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
        from utils.text_utils import normalize_text
        query_l = normalize_text(query)
        # Sumar keywords globales y del mapping
        rule_keywords = set(self.config.get('triggers', {}).get('keywords', []))
        for intent, cfg in CALENDAR_INTENT_MAPPING.items():
            for k in cfg.get('keywords', []):
                rule_keywords.add(k)
        hits = sum(1 for k in rule_keywords if k in query_l)
        return min(1.0, 0.15 * hits) if hits else 0.0

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        score = 0.0 if not self.service else self._rule_score(query)
        # Heurística: si el contexto previo es calendario, subir score
        last_qt = (context or {}).get('last_query_type', '')
        if isinstance(last_qt, str) and last_qt.startswith('calendario'):
            score = max(score, 0.7)
        return Decision(score=score, params={}, reasons=["calendar_rule_score"])

    def accepts(self, score: float) -> bool:
        accept = float(self.config.get('thresholds', {}).get('accept', 0.65))
        return score >= accept

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        if not self.service:
            return ToolResult(response="", sources=[], metadata={})

        # Intento de detectar intent por keywords del mapping
        intent = None
        from utils.text_utils import normalize_text
        ql = normalize_text(query)
        for candidate_intent, cfg in CALENDAR_INTENT_MAPPING.items():
            if any(k in ql for k in cfg.get('keywords', [])):
                intent = candidate_intent
                break

        text = get_calendar_events(self.service, intent)
        if not text:
            return ToolResult(response="", sources=[], metadata={})

        return ToolResult(response=text, sources=["Calendario Académico"], metadata={})


