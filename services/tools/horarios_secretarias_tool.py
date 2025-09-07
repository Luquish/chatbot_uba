import logging
import os
from typing import Any, Dict, List, Optional
from utils.text_utils import normalize_text as norm, detect_day_token

from .base import BaseTool, Decision, ToolResult
from services.sheets_service import SheetsService


logger = logging.getLogger(__name__)


class HorariosSecretariasTool:
    name = "sheets.horarios_secretarias"
    priority = 64

    def __init__(self, sheets_service: Optional[SheetsService]):
        self.sheets_service = sheets_service
        self.config: Dict[str, Any] = {
            'thresholds': { 'accept': 0.6 },
            'triggers': {
                'keywords': [
                    'secretaria', 'secretaría', 'alumnos', 'admisión', 'admision', 'tesoreria', 'tesorería',
                    'kinesiologia', 'obstetricia', 'hemoterapia', 'podologia', 'radiologia', 'naon',
                    'archivos', 'actas', 'pases', 'discapacidad', 'genero', 'género', 'seube', 'horario', 'mail', 'ubicacion', 'piso'
                ]
            },
            'spreadsheet_id': None,
            'sheet_name': 'Hoja 1',
            'ranges': { 'default': 'A:J' }
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
            return Decision(score=0.0, params={}, reasons=["secretarias_unavailable"])
        score = self._rule_score(query)
        return Decision(score=score, params={}, reasons=["secretarias_rule_score"])

    def accepts(self, score: float) -> bool:
        return score >= float(self.config.get('thresholds', {}).get('accept', 0.6))

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        sid = self.config.get('spreadsheet_id')
        if not sid:
            return ToolResult(response="", sources=[], metadata={})

        sheet_name = self.config.get('sheet_name', 'Hoja 1')
        rng = self.config.get('ranges', {}).get('default', 'A:J')
        a1 = f"'{sheet_name}'!{rng}"
        values = self.sheets_service.get_sheet_values(sid, a1)
        if not values or len(values) < 2:
            return ToolResult(response="", sources=[], metadata={})

        # Estructura esperada: A: NOMBRE SECRETARIA, B-F: L M W J V (marcas), G: HORARIO, H: SECTOR, I: PISO, J: MAIL
        header_idx = 0
        data_start = header_idx + 1
        rows = values[data_start:]

        ql = norm(query)

        # Utilidades de interpretación
        def detect_day(q: str):
            return detect_day_token(q)

        ask_mail = any(w in ql for w in ['mail', 'correo', 'email'])
        ask_horario = any(w in ql for w in ['horario', 'hora', 'desde', 'hasta', 'atiende', 'abren', 'abre', 'cierra'])
        ask_ubicacion = any(w in ql for w in ['ubicacion', 'ubicación', 'donde', 'dónde', 'sector'])
        ask_piso = 'piso' in ql
        ask_dias = any(w in ql for w in ['dias', 'días', 'que dia', 'que dias', 'qué dia', 'qué dias'])
        day_requested = detect_day(ql)

        # Alias comunes para mejorar el match
        alias = {
            'admision': 'admisión',
            'tesoreria': 'tesorería',
            'radiologia': 'radiología',
            'genero': 'género',
        }

        stop = {'sec', 'se', 'secretaria', 'secretaria de', 'secretaria del', 'secretaria de la', 'secretaria de el'}

        def name_score(name: str) -> int:
            n = norm(name)
            n = alias.get(n, n)
            tokens = [t for t in n.split() if t not in stop]
            return sum(1 for t in tokens if t in ql)

        def day_flag(v: Any) -> str:
            return "✔" if str(v).strip() else "—"

        # Buscar mejor coincidencia por nombre si la consulta sugiere una secretaría
        best_row: Optional[List[Any]] = None
        best_score = 0
        for r in rows:
            if not r or len(r) < 10:
                continue
            name = str(r[0]).strip()
            sc = name_score(name)
            if sc > best_score:
                best_score = sc
                best_row = r

        # Funciones de formato
        def format_days(Ld: str, Md: str, Wd: str, Jd: str, Vd: str) -> str:
            return f"L:{Ld} M:{Md} W:{Wd} J:{Jd} V:{Vd}"

        def day_flag(v: Any) -> str:
            return "✔" if str(v).strip() else "—"

        # Si hay una secretaría objetivo
        if best_row and best_score > 0:
            name = str(best_row[0]).strip()
            Ld = day_flag(best_row[1] if len(best_row) > 1 else "")
            Md = day_flag(best_row[2] if len(best_row) > 2 else "")
            Wd = day_flag(best_row[3] if len(best_row) > 3 else "")
            Jd = day_flag(best_row[4] if len(best_row) > 4 else "")
            Vd = day_flag(best_row[5] if len(best_row) > 5 else "")
            horario = str(best_row[6]).strip() if len(best_row) > 6 else ""
            sector = str(best_row[7]).strip() if len(best_row) > 7 else ""
            piso = str(best_row[8]).strip() if len(best_row) > 8 else ""
            mail = str(best_row[9]).strip() if len(best_row) > 9 else ""

            # Filtros específicos
            if day_requested:
                map_back = {'L': 'lunes', 'M': 'martes', 'W': 'miércoles', 'J': 'jueves', 'V': 'viernes'}
                open_flag = {
                    'L': Ld, 'M': Md, 'W': Wd, 'J': Jd, 'V': Vd
                }.get(day_requested, '—')
                if open_flag == '✔':
                    return ToolResult(
                        response=f"✅ {name} atiende el {map_back.get(day_requested)}. HORARIO: {horario}",
                        sources=["Google Sheet Horarios Secretarías"],
                        metadata={"name": name, "day": day_requested, "open": True}
                    )
                else:
                    return ToolResult(
                        response=f"⚠️ {name} NO atiende el {map_back.get(day_requested)}.",
                        sources=["Google Sheet Horarios Secretarías"],
                        metadata={"name": name, "day": day_requested, "open": False}
                    )

            # Respuestas enfocadas
            if ask_mail and not (ask_horario or ask_ubicacion or ask_piso or ask_dias):
                return ToolResult(response=f"✉️ {name}: {mail}", sources=["Google Sheet Horarios Secretarías"], metadata={"name": name, "mail": mail})
            if ask_ubicacion and not (ask_mail or ask_piso or ask_horario or ask_dias):
                return ToolResult(response=f"📍 {name}: SECTOR {sector}", sources=["Google Sheet Horarios Secretarías"], metadata={"name": name, "sector": sector})
            if ask_piso and not (ask_mail or ask_ubicacion or ask_horario or ask_dias):
                return ToolResult(response=f"🏢 {name}: PISO {piso}", sources=["Google Sheet Horarios Secretarías"], metadata={"name": name, "piso": piso})
            if ask_horario and not (ask_mail or ask_ubicacion or ask_piso or ask_dias):
                return ToolResult(response=f"🕒 {name}: {horario}", sources=["Google Sheet Horarios Secretarías"], metadata={"name": name, "horario": horario})
            if ask_dias and not (ask_mail or ask_ubicacion or ask_piso or ask_horario):
                return ToolResult(response=f"📅 {name}: {format_days(Ld, Md, Wd, Jd, Vd)}", sources=["Google Sheet Horarios Secretarías"], metadata={"name": name})

            # Respuesta completa por defecto para la secretaría
            text = (
                f"{name}\n- DÍAS: {format_days(Ld, Md, Wd, Jd, Vd)}\n"
                f"- HORARIO: {horario}\n- SECTOR: {sector} | PISO: {piso}\n- MAIL: {mail}"
            )
            return ToolResult(response=text, sources=["Google Sheet Horarios Secretarías"], metadata={"name": name})

        # Si no hay una secretaría específica detectada, se puede listar según filtros globales (ej. por día)
        results: List[str] = []
        for r in rows:
            if not r or len(r) < 10:
                continue
            name = str(r[0]).strip()
            Ld = day_flag(r[1] if len(r) > 1 else "")
            Md = day_flag(r[2] if len(r) > 2 else "")
            Wd = day_flag(r[3] if len(r) > 3 else "")
            Jd = day_flag(r[4] if len(r) > 4 else "")
            Vd = day_flag(r[5] if len(r) > 5 else "")
            horario = str(r[6]).strip() if len(r) > 6 else ""
            sector = str(r[7]).strip() if len(r) > 7 else ""
            piso = str(r[8]).strip() if len(r) > 8 else ""
            mail = str(r[9]).strip() if len(r) > 9 else ""

            if day_requested:
                open_flag = {'L': Ld, 'M': Md, 'W': Wd, 'J': Jd, 'V': Vd}.get(day_requested, '—')
                if open_flag != '✔':
                    continue

            results.append(
                f"{name}\n- DÍAS: {format_days(Ld, Md, Wd, Jd, Vd)}\n- HORARIO: {horario}\n- SECTOR: {sector} | PISO: {piso}\n- MAIL: {mail}"
            )

        if not results:
            return ToolResult(response="", sources=[], metadata={})

        preview = "\n\n".join(results[:8])
        if len(results) > 8:
            preview += f"\n\n(+{len(results)-8} más)"

        return ToolResult(
            response=f"🏢 HORARIOS DE SECRETARÍAS\n\n{preview}",
            sources=["Google Sheet Horarios Secretarías"],
            metadata={}
        )


