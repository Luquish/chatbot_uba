import logging
import os
from typing import Any, Dict, List, Optional
from utils.text_utils import normalize_text as norm

from .base import BaseTool, Decision, ToolResult
from services.sheets_service import SheetsService


logger = logging.getLogger(__name__)


class HorariosLicTecTool:
    name = "sheets.horarios_lic_tec"
    priority = 63

    def __init__(self, sheets_service: Optional[SheetsService]):
        self.sheets_service = sheets_service
        self.config: Dict[str, Any] = {
            'thresholds': { 'accept': 0.6 },
            'triggers': {
                'keywords': [
                    "licenciatura", "tecnicatura", "lic", "tec",
                    "kinesiologia", "fonoaudiologia", "nutricion", "enfermeria",
                    "obstetricia", "bioimagenes", "podologia", "anestesia",
                    "cosmetologia", "hemoterapia", "instrumentacion",
                    "practicas cardiologicas", "atencion", "ubicacion", "contacto"
                ]
            },
            'spreadsheet_id': None,
            'sheet_name': 'Hoja 1',
            'ranges': { 'default': 'A:D' }
        }

    def configure(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        sid = config.get('spreadsheet_id')
        if isinstance(sid, str) and sid.startswith('ENV:'):
            env_key = sid.split(':', 1)[1]
            config['spreadsheet_id'] = os.getenv(env_key)
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        ql = norm(query)
        hits = sum(1 for k in self.config.get('triggers', {}).get('keywords', []) if k in ql)
        return min(1.0, 0.2 * hits) if hits else 0.0

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        if not self.sheets_service or not self.config.get('spreadsheet_id'):
            return Decision(score=0.0, params={}, reasons=["lic_tec_unavailable"])
        score = self._rule_score(query)
        return Decision(score=score, params={}, reasons=["lic_tec_rule_score"])

    def accepts(self, score: float) -> bool:
        return score >= float(self.config.get('thresholds', {}).get('accept', 0.6))

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        sid = self.config.get('spreadsheet_id')
        if not sid:
            return ToolResult(response="", sources=[], metadata={})

        sheet_name = self.config.get('sheet_name', 'Hoja 1')
        rng = self.config.get('ranges', {}).get('default', 'A:D')
        a1 = f"'{sheet_name}'!{rng}"
        values = self.sheets_service.get_sheet_values(sid, a1)
        if not values or len(values) < 2:
            return ToolResult(response="", sources=[], metadata={})

        header_idx = 0  # Fila 1 es cabecera
        data_start = header_idx + 1
        rows = values[data_start:]

        ql = norm(query)
        # Col A: NOMBRE (LIC/TEC), B: CONTACTO (email), C: UBICACIÓN, D: ATENCIÓN
        def match_program(name: str) -> bool:
            n = norm(name)
            tokens = n.split()
            return n in ql or any(t in ql for t in tokens)

        results: List[str] = []
        for r in rows:
            if not r or len(r) < 4:
                continue
            program = str(r[0]).strip()
            contact = str(r[1]).strip()
            location = str(r[2]).strip()
            attention = str(r[3]).strip()

            if not program:
                continue

            if any(k in ql for k in ["lic", "licenciatura", "tec", "tecnicatura"]) or match_program(program):
                results.append(
                    f"{program}\n- CONTACTO: {contact}\n- UBICACION: {location}\n- ATENCION: {attention}"
                )

        if not results:
            return ToolResult(response="", sources=[], metadata={})

        preview = "\n\n".join(results[:8])
        if len(results) > 8:
            preview += f"\n\n(+{len(results)-8} más)"

        return ToolResult(
            response=f"🎓 HORARIOS LICENCIATURAS Y TECNICATURAS\n\n{preview}",
            sources=["Google Sheet Horarios LIC y TEC"],
            metadata={}
        )


