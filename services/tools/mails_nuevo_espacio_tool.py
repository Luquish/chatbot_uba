import logging
import os
from typing import Any, Dict, List, Optional
from utils.text_utils import normalize_text as norm

from .base import BaseTool, Decision, ToolResult
from services.sheets_service import SheetsService


logger = logging.getLogger(__name__)


class MailsNuevoEspacioTool:
    name = "sheets.mails_nuevo_espacio"
    priority = 62

    def __init__(self, sheets_service: Optional[SheetsService]):
        self.sheets_service = sheets_service
        self.config: Dict[str, Any] = {
            'thresholds': { 'accept': 0.6 },
            'triggers': {
                'keywords': [
                    'voluntariado', 'imprimir', 'impresion', 'impresiones', 'histoteca',
                    'readmision', 'reincorporacion', 'recursada', 'prorroga', 'prórroga', 'nuevo espacio'
                ]
            },
            'spreadsheet_id': None,
            'sheet_name': 'Hoja 1',
            'ranges': { 'default': 'A:A' }
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
        return min(1.0, 0.25 * hits) if hits else 0.0

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        if not self.sheets_service or not self.config.get('spreadsheet_id'):
            return Decision(score=0.0, params={}, reasons=["mails_unavailable"])
        score = self._rule_score(query)
        return Decision(score=score, params={}, reasons=["mails_rule_score"])

    def accepts(self, score: float) -> bool:
        return score >= float(self.config.get('thresholds', {}).get('accept', 0.6))

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        sid = self.config.get('spreadsheet_id')
        if not sid:
            return ToolResult(response="", sources=[], metadata={})

        a1 = f"'{self.config.get('sheet_name', 'Hoja 1')}'!{self.config.get('ranges', {}).get('default', 'A:A')}"
        values = self.sheets_service.get_sheet_values(sid, a1)
        if not values or len(values) < 2:
            return ToolResult(response="", sources=[], metadata={})

        ql = norm(query)

        # Mapea tipo → email por posición (filas 2..6)
        emails: List[str] = []
        for row in values[1:]:
            if row and row[0]:
                emails.append(str(row[0]).strip())

        # Indices esperados
        mail_voluntariado = emails[0] if len(emails) > 0 else ""
        mail_nuevoespacio = emails[1] if len(emails) > 1 else ""
        mail_impresiones = emails[2] if len(emails) > 2 else ""
        mail_histoteca = emails[3] if len(emails) > 3 else ""
        mail_readmision = emails[4] if len(emails) > 4 else ""

        # Política de derivación: para (2) no damos email, derivamos a Instagram
        instagram = "@cecim.nemed"

        # Clasificar intención
        if 'voluntari' in ql or 'sumarme' in ql:
            return ToolResult(response=f"✉️ Voluntariado: {mail_voluntariado}", sources=["Mails Nuevo Espacio"], metadata={"type": "voluntariado", "mail": mail_voluntariado})

        if 'imprim' in ql or 'impresion' in ql or 'impresiones' in ql:
            return ToolResult(response=f"🖨️ Impresiones: {mail_impresiones}", sources=["Mails Nuevo Espacio"], metadata={"type": "impresiones", "mail": mail_impresiones})

        if 'histoteca' in ql:
            return ToolResult(response=f"🧪 Histoteca: {mail_histoteca}", sources=["Mails Nuevo Espacio"], metadata={"type": "histoteca", "mail": mail_histoteca})

        if any(k in ql for k in ['recursad', 'prorrog', 'readmision', 'reincorporac']):
            return ToolResult(response=f"📩 Readmisión/Recursada/Prórrogas: {mail_readmision}", sources=["Mails Nuevo Espacio"], metadata={"type": "readmision", "mail": mail_readmision})

        # Cualquier otro caso relacionado con “nuevo espacio” o dudas genéricas → derivación a IG
        if 'nuevo espacio' in ql or 'nuevoespacio' in ql or 'contact' in ql:
            return ToolResult(response=f"📱 Para consultas generales, escribinos a Instagram: {instagram}", sources=["Mails Nuevo Espacio"], metadata={"type": "instagram", "handle": instagram})

        return ToolResult(response="", sources=[], metadata={})


