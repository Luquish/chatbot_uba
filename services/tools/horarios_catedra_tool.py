import logging
import os
from typing import Any, Dict, List, Optional
import re
from utils.text_utils import normalize_text as norm, extract_catedra_number, is_count_catedras_query

from .base import BaseTool, Decision, ToolResult, SheetsBaseTool, MatchDetails
from services.sheets_service import SheetsService


logger = logging.getLogger(__name__)


class HorariosCatedraTool(SheetsBaseTool):
    name = "sheets.horarios_catedra"
    priority = 65

    def __init__(self, sheets_service: Optional[SheetsService]):
        # Configuración específica para horarios de cátedra
        default_config = {
            'thresholds': {'accept': 0.62},
            'triggers': {
                'keywords': [
                    "horario", "cátedra", "catedra", "materia", "secretaría", "secretaria", 
                    "sector", "piso", "mail", "clase", "clases", "profesor", "profesores",
                    "aula", "aulas", "turno", "turnos", "mañana", "tarde", "noche",
                    "lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"
                ]
            },
            'spreadsheet_id': None,
            'sheet_name': 'Hoja 1',
            'ranges': {'default': 'A:M'},
            'caching': {
                'enabled': True,
                'ttl_minutes': 15  # Caché más corto para datos dinámicos
            },
            'fuzzy_matching': {
                'enabled': True,
                'threshold': 0.6,
                'weights': {
                    'ratio': 0.3,
                    'partial': 0.25,
                    'token_sort': 0.25,
                    'token_set': 0.2
                }
            },
            'data_processing': {
                'skip_header_rows': 2,  # Saltar 2 filas de header
                'normalize_columns': True,
                'handle_merged_cells': True
            }
        }
        
        # Usar constructor de SheetsBaseTool
        super().__init__(self.name, self.priority, sheets_service, default_config)

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
        # Usar función heredada de ModernBaseTool
        base_score = super()._rule_score(query)
        
        # Boost por días específicos (lógica específica de cátedra)
        ql = self._normalize_query(query)
        day_boost = 0.0
        if any(d in ql for d in ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo", "l ", " m", " w", " j", " v"]):
            day_boost = 0.2
        
        return min(base_score + day_boost, 1.0)

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
        try:
            logger.info(f"Procesando consulta de horarios de cátedra: {query}")
            
            # Usar función heredada para obtener datos del sheet
            values = self._fetch_sheet_data()
            if not values:
                return ToolResult(response="", sources=[], metadata={})

            # Usar función heredada para procesar filas
            rows = self._process_sheet_rows(values)

            # Usar función heredada para normalizar consulta
            ql = self._normalize_query(query)
            
            # Filtrado mejorado por materia y/o cátedra usando fuzzy matching
            materia_filter: Optional[str] = None
            catedra_filter: Optional[str] = None
            
            # Detectar números de cátedra en texto
            catedra_filter = extract_catedra_number(ql)
            
            # Detectar materia usando fuzzy matching
            materias_seen = set()
            for r in rows:
                if len(r) > 0 and r[0]:
                    materias_seen.add(str(r[0]).strip())
            
            # Buscar mejor coincidencia de materia usando fuzzy matching
            best_materia_score = 0.0
            for materia in materias_seen:
                if materia:
                    score, _ = self._calculate_fuzzy_score(ql, materia)
                    if score > best_materia_score and score > 0.6:
                        best_materia_score = score
                        materia_filter = materia

            # Si la consulta pregunta "cuántas cátedras" -> responder conteo por materia
            if materia_filter and is_count_catedras_query(ql):
                catedras: List[str] = []
                for r in rows:
                    if not r or len(r) < 2:
                        continue
                    materia = self._normalize_query(str(r[0]).strip())
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

                if materia_filter and self._normalize_query(materia) != materia_filter:
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

            logger.info(f"Consulta de horarios de cátedra completada: {len(matches)} resultados")
            return ToolResult(
                response=f"📋 HORARIOS DE CÁTEDRA\n\n{preview}",
                sources=["Google Sheet Horarios Cátedra"],
                metadata={"matches_count": len(matches), "materia_filter": materia_filter, "catedra_filter": catedra_filter}
            )
            
        except Exception as e:
            logger.error(f"Error procesando consulta de horarios de cátedra: {e}")
            return ToolResult(
                response="",
                sources=[],
                metadata={'error': str(e)}
            )


