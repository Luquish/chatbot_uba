import logging
import os
from typing import Any, Dict, List, Optional
import re
from utils.text_utils import normalize_text as norm, extract_catedra_number, is_count_catedras_query

from .base import BaseTool, Decision, ToolResult
from services.sheets_service import SheetsService


logger = logging.getLogger(__name__)


class HorariosCatedraTool:
    name = "sheets.horarios_catedra"
    priority = 65

    def __init__(self, sheets_service: Optional[SheetsService]):
        self.sheets_service = sheets_service
        self.config: Dict[str, Any] = {
            'thresholds': { 'accept': 0.62 },
            'triggers': { 'keywords': ["horario", "cátedra", "catedra", "materia", "secretaría", "secretaria", "sector", "piso", "mail"] },
            'spreadsheet_id': None,
            'sheet_name': 'Hoja 1',
            'ranges': { 'default': 'A:M' }
        }

    def configure(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        # Resolver ENV:VAR si aplica
        sid = config.get('spreadsheet_id')
        if isinstance(sid, str) and sid.startswith('ENV:'):
            env_key = sid.split(':', 1)[1]
            config['spreadsheet_id'] = os.getenv(env_key)
        self.config.update(config)

    def _rule_score(self, query: str) -> float:
        ql = norm(query)
        hits = sum(1 for k in self.config.get('triggers', {}).get('keywords', []) if k in ql)
        # Si menciona días específicos, subir score
        if any(d in ql for d in ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo", "l ", " m", " w", " j", " v"]):
            hits += 1
        return min(1.0, 0.18 * hits) if hits else 0.0

    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        if not self.sheets_service or not self.config.get('spreadsheet_id'):
            return Decision(score=0.0, params={}, reasons=["horarios_unavailable"])
        score = self._rule_score(query)
        # Si el contexto previo fue cursos/horarios, subir score
        last_qt = (context or {}).get('last_query_type', '')
        if last_qt in ('cursos', 'horarios_catedra'):
            score = max(score, 0.7)
        return Decision(score=score, params={}, reasons=["horarios_rule_score"])

    def accepts(self, score: float) -> bool:
        return score >= float(self.config.get('thresholds', {}).get('accept', 0.62))

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        sid = self.config.get('spreadsheet_id')
        if not sid:
            return ToolResult(response="", sources=[], metadata={})

        sheet_name = self.config.get('sheet_name', 'Hoja 1')
        rng = self.config.get('ranges', {}).get('default', 'A:M')
        a1 = f"'{sheet_name}'!{rng}"
        values = self.sheets_service.get_sheet_values(sid, a1)
        if not values:
            return ToolResult(response="", sources=[], metadata={})

        # Mapear columnas por posición (según estructura confirmada)
        # A MATERIA, B CATEDRA, C SECTOR, D PISO, E..I días L M W J V, J HORARIO, K SECTOR (sec), L PISO (sec), M MAIL
        header_row_idx = 1  # fila 2 (0-based indexing)
        data_start_idx = header_row_idx + 1
        raw_rows = values[data_start_idx:]

        # Rellenar celdas combinadas de MATERIA hacia abajo para no perder el contexto
        rows: List[List[Any]] = []
        current_materia: Optional[str] = None
        for r in raw_rows:
            if not r:
                rows.append(r)
                continue
            materia_cell = str(r[0]).strip() if len(r) > 0 and r[0] else ""
            if materia_cell:
                current_materia = materia_cell
            else:
                # Propagar la última materia vista
                if current_materia:
                    # Asegurar longitud suficiente
                    r = list(r)
                    while len(r) < 13:
                        r.append("")
                    r[0] = current_materia
            rows.append(r)

        ql = norm(query)
        # Filtrado simple por materia y/o cátedra si aparecen en la consulta
        materia_filter: Optional[str] = None
        catedra_filter: Optional[str] = None
        # Detectar números de cátedra en texto
        catedra_filter = extract_catedra_number(ql)
        # Detectar materia por palabras antes conocidas dentro de la hoja
        # Heurística: buscar tokens en la columna A presentes en la consulta (en mayúsculas en el sheet)
        materias_seen = set()
        for r in rows:
            if len(r) > 0 and r[0]:
                materias_seen.add(norm(str(r[0]).strip()))
        for mt in sorted(materias_seen, key=len, reverse=True):
            if mt and mt in ql:
                materia_filter = mt
                break

        # Si la consulta pregunta "cuántas cátedras" -> responder conteo por materia
        if materia_filter and is_count_catedras_query(ql):
            catedras: List[str] = []
            for r in rows:
                if not r or len(r) < 2:
                    continue
                materia = norm(str(r[0]).strip())
                if materia != materia_filter:
                    continue
                catedra = str(r[1]).strip()
                if not catedra or catedra.lower() == 'todas':
                    continue
                if catedra not in catedras:
                    catedras.append(catedra)

            if catedras:
                lista = ", ".join(catedras)
                nombre_materia = materia_filter.upper()
                return ToolResult(
                    response=f"📚 {nombre_materia}: hay {len(catedras)} cátedras (" + lista + ")",
                    sources=["Google Sheet Horarios Cátedra"],
                    metadata={"materia": nombre_materia, "count": len(catedras), "catedras": catedras}
                )
            else:
                return ToolResult(response="", sources=[], metadata={})

        # Construir resultados compactos
        def day_flag(v: Any) -> str:
            return "✔" if (str(v).strip() != "") else "—"

        matches: List[str] = []
        for r in rows:
            if not r or len(r) < 13:
                continue
            materia = str(r[0]).strip() if len(r) > 0 else ""
            catedra = str(r[1]).strip() if len(r) > 1 else ""
            sector = str(r[2]).strip() if len(r) > 2 else ""
            piso = str(r[3]).strip() if len(r) > 3 else ""
            Ld = day_flag(r[4] if len(r) > 4 else "")
            Md = day_flag(r[5] if len(r) > 5 else "")
            Wd = day_flag(r[6] if len(r) > 6 else "")
            Jd = day_flag(r[7] if len(r) > 7 else "")
            Vd = day_flag(r[8] if len(r) > 8 else "")
            horario = str(r[9]).strip() if len(r) > 9 else ""
            sec_sector = str(r[10]).strip() if len(r) > 10 else ""
            sec_piso = str(r[11]).strip() if len(r) > 11 else ""
            mail = str(r[12]).strip() if len(r) > 12 else ""

            if materia_filter and norm(materia) != materia_filter:
                continue
            if catedra_filter and catedra_filter != catedra:
                continue

            matches.append(
                f"MATERIA: {materia} | CÁTEDRA: {catedra}\n"
                f"- SECTOR: {sector} | PISO: {piso}\n"
                f"- DÍAS: L:{Ld} M:{Md} W:{Wd} J:{Jd} V:{Vd}\n"
                f"- SECRETARÍA: {horario} | SECTOR: {sec_sector} | PISO: {sec_piso}\n"
                f"- MAIL: {mail}"
            )

        if not matches:
            return ToolResult(response="", sources=[], metadata={})

        # Limitar salida para no saturar
        preview = "\n\n".join(matches[:5])
        if len(matches) > 5:
            preview += f"\n\n(+{len(matches)-5} más)"

        return ToolResult(
            response=f"📋 HORARIOS DE CÁTEDRA\n\n{preview}",
            sources=["Google Sheet Horarios Cátedra"],
            metadata={}
        )


