import logging
import os
from typing import Any, Dict, List, Optional

from .base import BaseTool, Decision, ToolResult
from services.sheets_service import SheetsService
from handlers.courses_handler import handle_sheet_course_query


logger = logging.getLogger(__name__)


class SheetsTool:
    name = "sheets.cursos"
    priority = 60

    def __init__(self, sheets_service: Optional[SheetsService], date_utils):
        self.sheets_service = sheets_service
        self.date_utils = date_utils
        self.config: Dict[str, Any] = {
            'thresholds': {
                'accept': 0.6
            },
            'triggers': {
                'keywords': ['curso', 'cursada', 'horario', 'clases']
            },
            'spreadsheet_id': os.getenv('CURSOS_SPREADSHEET_ID')
        }

    def configure(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        # Soporte para 'ENV:VAR'
        env_id = config.get('spreadsheet_id')
        if isinstance(env_id, str) and env_id.startswith('ENV:'):
            env_key = env_id.split(':', 1)[1]
            config['spreadsheet_id'] = os.getenv(env_key)
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        query_l = query.lower()
        keywords: List[str] = self.config.get('triggers', {}).get('keywords', [])
        hits = sum(1 for k in keywords if k in query_l)
        return min(1.0, 0.2 * hits) if hits else 0.0

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        if not self.sheets_service or not self.config.get('spreadsheet_id'):
            return Decision(score=0.0, params={}, reasons=["sheets_unavailable"])
        score = self._rule_score(query)
        # Si el contexto previo fue cursos, subir score
        last_qt = (context or {}).get('last_query_type', '')
        if last_qt == 'cursos':
            score = max(score, 0.7)
        return Decision(score=score, params={}, reasons=["sheets_rule_score"])

    def accepts(self, score: float) -> bool:
        accept = float(self.config.get('thresholds', {}).get('accept', 0.6))
        return score >= accept

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        spreadsheet_id = self.config.get('spreadsheet_id')
        if not spreadsheet_id:
            return ToolResult(response="", sources=[], metadata={})
        text = handle_sheet_course_query(query, self.sheets_service, spreadsheet_id, self.date_utils)
        if not text:
            return ToolResult(response="", sources=[], metadata={})
        return ToolResult(response=text, sources=["Google Sheet Cursos"], metadata={})


