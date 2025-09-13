import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
from utils.text_utils import normalize_text as norm, detect_day_token, clean_query_for_embedding
import re

# Librerías modernas para fuzzy matching
from rapidfuzz import fuzz, process
from fuzzywuzzy import fuzz as fw_fuzz

from .base import BaseTool, Decision, ToolResult, SheetsBaseTool, MatchDetails
from services.sheets_service import SheetsService


logger = logging.getLogger(__name__)


class SecretariaMatchError(Exception):
    """Excepción para errores de matching de secretarías."""
    pass

class SecretariaDataError(Exception):
    """Excepción para errores de datos de secretarías."""
    pass

class HorariosSecretariasTool(SheetsBaseTool):
    name = "sheets.horarios_secretarias"
    priority = 75

    def __init__(self, sheets_service: Optional[SheetsService]):
        # Configuración específica para secretarías
        default_config = {
            'thresholds': {'accept': 0.4},
            'triggers': {
                'keywords': [
                    # Keywords básicas
                    'secretaria', 'secretaría', 'alumnos', 'admisión', 'admision', 'tesoreria', 'tesorería',
                    'kinesiologia', 'obstetricia', 'hemoterapia', 'podologia', 'radiologia', 'naon',
                    'archivos', 'actas', 'pases', 'discapacidad', 'genero', 'género', 'seube', 
                    'horario', 'mail', 'ubicacion', 'piso',
                    # Keywords expandidas
                    'catedra', 'cátedra', 'departamento', 'depto', 'ciclo', 'biomedico', 'biomédico',
                    'clinico', 'clínico', 'anatomia', 'anatomía', 'histologia', 'histología',
                    'embriologia', 'embriología', 'fisiologia', 'fisiología', 'bioquimica', 'bioquímica',
                    'microbiologia', 'microbiología', 'farmacologia', 'farmacología', 'toxicologia',
                    'toxicología', 'patologia', 'patología', 'inmunologia', 'inmunología', 'bioetica',
                    'bioética', 'salud mental', 'medicina legal', 'salud publica', 'salud pública',
                    'familiar', 'enfermeria', 'enfermería', 'fonoaudiologia', 'fonoaudiología',
                    'nutricion', 'nutrición', 'cosmetologia', 'cosmetología', 'prac cardio',
                    'museo', 'uba xxi', 'actas certificaciones', 'títulos', 'graduados', 'posgrados',
                    'conexas', 'simultaneidad', 'ciclo biomedico', 'ciclo clinico'
                ]
            },
            'spreadsheet_id': None,
            'sheet_name': 'Hoja 1',
            'ranges': {'default': 'A:J'},
            'caching': {
                'enabled': True,
                'ttl_minutes': 30  # 30 minutos de caché
            },
            'fuzzy_matching': {
                'enabled': True,
                'threshold': 0.7,
                'weights': {
                    'ratio': 0.3,
                    'partial': 0.25,
                    'token_sort': 0.25,
                    'token_set': 0.2
                }
            }
        }
        
        # Usar constructor de SheetsBaseTool
        super().__init__(self.name, self.priority, sheets_service, default_config)
        
        # Alias para mejorar matching
        self.alias_map = {
            'admision': 'admisión',
            'tesoreria': 'tesorería',
            'radiologia': 'radiología',
            'genero': 'género',
            'anatomia': 'anatomía',
            'catedra': 'cátedra',
            'histologia': 'histología',
            'embriologia': 'embriología',
            'fisiologia': 'fisiología',
            'bioquimica': 'bioquímica',
            'farmacologia': 'farmacología',
            'microbiologia': 'microbiología',
            'toxicologia': 'toxicología',
            'patologia': 'patología',
            'inmunologia': 'inmunología'
        }
        
        # Stop words para limpiar consultas
        self.stop_words = {
            'sec', 'se', 'secretaria', 'secretaría', 'de', 'del', 'la', 'el', 'donde', 'dónde', 
            'está', 'esta', 'queda', 'catedra', 'cátedra', 'todas', 'todos'
        }

    def configure(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        sid = config.get('spreadsheet_id')
        if isinstance(sid, str) and sid.startswith('ENV:'):
            env_key = sid.split(':', 1)[1]
            config['spreadsheet_id'] = os.getenv(env_key)
        self.config.update(config)

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto aplicando alias y limpieza mejorada."""
        # Usar función heredada de ModernBaseTool
        normalized = self._normalize_query(text)
        
        # Aplicar alias específicos de secretarías
        for alias, canonical in self.alias_map.items():
            normalized = normalized.replace(alias, canonical)
        
        # Limpieza adicional para embeddings
        normalized = clean_query_for_embedding(normalized)
        
        return normalized
    
    def _extract_query_components(self, query: str) -> Dict[str, Any]:
        """Extrae componentes semánticos de la consulta."""
        normalized = self._normalize_text(query)
        
        # Patrones para extraer información específica
        patterns = {
            'catedra_num': r'c[aá]tedra\s*(\d+)',
            'ciclo': r'(biomedico|biom[eé]dico|clinico|cl[ií]nico)',
            'area': r'(anatomia|anatomía|histologia|histología|embriologia|embriología|'
                    r'fisiologia|fisiología|bioquimica|bioquímica|microbiologia|microbiología|'
                    r'farmacologia|farmacología|toxicologia|toxicología|patologia|patología|'
                    r'inmunologia|inmunología|bioetica|bioética|salud\s*mental|medicina\s*legal|'
                    r'salud\s*publica|salud\s*pública)',
            'departamento': r'(alumnos?|admisi[oó]n|tesorer[ií]a|t[ií]tulos?|graduados?|posgrados?|'
                           r'conexas?|archivos?|actas?|certificaciones?|pases?|simultaneidad)',
            'especialidad': r'(enfermer[ií]a|obstetricia|hemoterapia|podolog[ií]a|radiolog[ií]a|'
                           r'kinesiolog[ií]a|nutrici[oó]n|cosmetolog[ií]a|fonoaudiolog[ií]a)',
            'especial': r'(museo\s*na[oó]n|uba\s*xxi|seube|depto\.\s*discapacidad|depto\.\s*g[eé]nero)'
        }
        
        components = {
            'original': query,
            'normalized': normalized,
            'catedra_num': None,
            'area': None,
            'departamento': None,
            'especialidad': None,
            'especial': None,
            'ciclo': None
        }
        
        # Extraer componentes usando regex
        for key, pattern in patterns.items():
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                if key == 'catedra_num':
                    components[key] = match.group(1)
                else:
                    components[key] = match.group(0)
        
        # Extraer keywords restantes
        words_to_remove = ['secretaria', 'secretaría', 'donde', 'dónde', 'esta', 'está', 
                          'queda', 'de', 'del', 'la', 'el', 'los', 'las']
        words = [w for w in normalized.split() if w not in words_to_remove and len(w) > 2]
        components['keywords'] = words
        
        return components
    
    def _calculate_match_score(self, query_components: Dict[str, Any], secretaria_name: str) -> Tuple[float, MatchDetails]:
        """Calcula un score sofisticado de matching usando funciones heredadas."""
        query_normalized = query_components['normalized']
        
        # Usar función heredada de ModernBaseTool
        score, match_details = self._calculate_fuzzy_score(query_normalized, secretaria_name)
        
        # Si hay coincidencia exacta, retornar inmediatamente
        if match_details.exact_match:
            return score, match_details
        
        # 3. Matching por componentes específicos (mejorado)
        name_normalized = self._normalize_text(secretaria_name)
        component_weights = {
            'area': 0.4,
            'catedra_num': 0.3,
            'departamento': 0.5,
            'especialidad': 0.5,
            'especial': 0.6,
            'ciclo': 0.4
        }
        
        for component, weight in component_weights.items():
            if query_components[component]:
                component_value = query_components[component]
                if component_value in name_normalized:
                    score += weight
                    match_details.component_matches[component] = True
                elif component == 'catedra_num':
                    # Buscar número de cátedra en el nombre
                    if f"catedra {component_value}" in name_normalized or f"cátedra {component_value}" in name_normalized:
                        score += weight
                        match_details.component_matches[component] = True
        
        # 4. Fuzzy matching mejorado para keywords individuales
        if query_components['keywords']:
            keyword_scores = []
            for keyword in query_components['keywords']:
                # Usar función heredada para cada keyword
                keyword_score, _ = self._calculate_fuzzy_score(keyword, name_normalized)
                
                if keyword_score > 0.6:  # Threshold más bajo para capturar más coincidencias
                    keyword_scores.append(keyword_score)
                    match_details.keyword_matches.append(f"{keyword} (score: {keyword_score:.2f})")
            
            if keyword_scores:
                avg_keyword_score = sum(keyword_scores) / len(keyword_scores)
                # Ponderar por cantidad de keywords que matchearon
                match_ratio = len(keyword_scores) / len(query_components['keywords'])
                score += (avg_keyword_score * match_ratio) * 0.3
        
        # 5. Penalización si hay componentes que NO coinciden
        if query_components['catedra_num'] and 'catedra' in name_normalized.lower():
            # Si pide cátedra específica pero no coincide el número
            catedra_pattern = r'c[aá]tedra\s*(\d+)'
            name_match = re.search(catedra_pattern, name_normalized, re.IGNORECASE)
            if name_match and name_match.group(1) != query_components['catedra_num']:
                score *= 0.3  # Penalización fuerte
        
        # 6. Si no hay score por componentes, usar fuzzy score como fallback
        if score == 0.0 and match_details.fuzzy_scores.get('combined', 0) > 0.3:
            score = match_details.fuzzy_scores['combined'] * 0.8  # Reducir un poco para dar prioridad a matches específicos
        
        return min(score, 1.0), match_details
    
    def _rule_score(self, query: str) -> float:
        """Calcula score basado en keywords de la consulta."""
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

    def _fetch_sheet_data(self) -> List[List[str]]:
        """Obtiene datos del sheet usando funcionalidad heredada."""
        # Usar función heredada de SheetsBaseTool
        values = super()._fetch_sheet_data()
        
        if not values or len(values) < 2:
            raise SecretariaDataError("No hay datos en el sheet o datos insuficientes")
        
        return values[1:]  # Omitir header

    def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Parsea la consulta para extraer la intención del usuario."""
        ql = norm(query)
        
        # Detectar tipo de información solicitada
        ask_mail = any(w in ql for w in ['mail', 'correo', 'email'])
        ask_horario = any(w in ql for w in ['horario', 'hora', 'desde', 'hasta', 'atiende', 'abren', 'abre', 'cierra'])
        ask_ubicacion = any(w in ql for w in ['ubicacion', 'ubicación', 'donde', 'dónde', 'sector'])
        ask_piso = 'piso' in ql
        ask_dias = any(w in ql for w in ['dias', 'días', 'que dia', 'que dias', 'qué dia', 'qué dias'])
        
        # Detectar día específico
        day_requested = detect_day_token(ql)
        
        # Detectar si se solicitan todas las secretarías
        ask_all = any(w in ql for w in ['todas', 'todos', 'completa', 'lista'])
        
        # Obtener componentes semánticos
        components = self._extract_query_components(query)
        
        return {
            'ask_mail': ask_mail,
            'ask_horario': ask_horario,
            'ask_ubicacion': ask_ubicacion,
            'ask_piso': ask_piso,
            'ask_dias': ask_dias,
            'day_requested': day_requested,
            'ask_all': ask_all,
            'components': components
        }

    def _find_matching_secretarias(self, query_components: Dict[str, Any], rows: List[List[str]]) -> List[Tuple[float, List[str], Dict[str, Any]]]:
        """Encuentra secretarías que coincidan usando el sistema de scoring mejorado."""
        matches = []
        
        for row in rows:
            if not row or len(row) < 10:
                continue
                
            name = str(row[0]).strip()
            if not name:
                continue
            
            # Calcular score usando el sistema mejorado
            score, match_details = self._calculate_match_score(query_components, name)
            
            # Threshold más inteligente basado en el tipo de match
            if score > 0.15 or match_details.get('exact_match') or any(match_details.get('component_matches', {}).values()):
                matches.append((score, row, match_details))
                logger.info(f"Match encontrado: {name} (score: {score:.3f}, detalles: {match_details})")
        
        # Ordenar por score descendente
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches

    def _format_secretaria_info(self, row: List[str], intent: Dict[str, Any]) -> str:
        """Formatea la información de una secretaría según la intención."""
        name = str(row[0]).strip()
        
        # Extraer datos de la fila
        def day_flag(v: Any) -> str:
            return "✔" if str(v).strip() else "—"
        
        Ld = day_flag(row[1] if len(row) > 1 else "")
        Md = day_flag(row[2] if len(row) > 2 else "")
        Wd = day_flag(row[3] if len(row) > 3 else "")
        Jd = day_flag(row[4] if len(row) > 4 else "")
        Vd = day_flag(row[5] if len(row) > 5 else "")
        horario = str(row[6]).strip() if len(row) > 6 else ""
        sector = str(row[7]).strip() if len(row) > 7 else ""
        piso = str(row[8]).strip() if len(row) > 8 else ""
        mail = str(row[9]).strip() if len(row) > 9 else ""
        
        def format_days(Ld: str, Md: str, Wd: str, Jd: str, Vd: str) -> str:
            return f"L:{Ld} M:{Md} W:{Wd} J:{Jd} V:{Vd}"
        
        # Respuestas específicas según la intención
        if intent['day_requested']:
            map_back = {'L': 'lunes', 'M': 'martes', 'W': 'miércoles', 'J': 'jueves', 'V': 'viernes'}
            open_flag = {
                'L': Ld, 'M': Md, 'W': Wd, 'J': Jd, 'V': Vd
            }.get(intent['day_requested'], '—')
            
            if open_flag == '✔':
                return f"✅ {name} atiende el {map_back.get(intent['day_requested'])}. HORARIO: {horario}"
            else:
                return f"⚠️ {name} NO atiende el {map_back.get(intent['day_requested'])}."
        
        # Respuestas enfocadas en un tipo de información
        exclusive_asks = sum([intent['ask_mail'], intent['ask_horario'], intent['ask_ubicacion'], intent['ask_piso'], intent['ask_dias']])
        
        if exclusive_asks == 1:
            if intent['ask_mail']:
                return f"✉️ {name}: {mail}"
            elif intent['ask_ubicacion']:
                return f"📍 {name}: SECTOR {sector}"
            elif intent['ask_piso']:
                return f"🏢 {name}: PISO {piso}"
            elif intent['ask_horario']:
                return f"🕒 {name}: {horario}"
            elif intent['ask_dias']:
                return f"📅 {name}: {format_days(Ld, Md, Wd, Jd, Vd)}"
        
        # Respuesta completa por defecto
        return (
            f"{name}\n- DÍAS: {format_days(Ld, Md, Wd, Jd, Vd)}\n"
            f"- HORARIO: {horario}\n- SECTOR: {sector} | PISO: {piso}\n- MAIL: {mail}"
        )

    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """Ejecuta la búsqueda de secretarías con manejo robusto de errores y logging mejorado."""
        start_time = time.time()
        
        try:
            logger.info(f"Procesando consulta de secretarías: {query}")
            
            # 1. Obtener datos del sheet (con caché)
            rows = self._fetch_sheet_data()
            
            # 2. Parsear intención de la consulta
            intent = self._parse_query_intent(query)
            logger.debug(f"Intención parseada: {intent}")
            
            # 3. Verificar si se piden todas las secretarías explícitamente
            if intent['ask_all']:
                logger.info("Usuario pidió explícitamente todas las secretarías")
                result = self._format_all_secretarias(rows, intent)
                logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
                return result
            
            # 4. Buscar coincidencias usando el sistema de scoring mejorado
            matches = self._find_matching_secretarias(intent['components'], rows)
            
            if not matches:
                logger.warning(f"No se encontraron coincidencias para: {intent['components']}")
                # Si no hay matches y la consulta es muy genérica, devolver todas
                if not intent['components']['area'] and not intent['components']['departamento'] and not intent['components']['especialidad']:
                    result = self._format_all_secretarias(rows, intent)
                    logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
                    return result
                else:
                    # Si había componentes específicos pero no matchearon, informar error
                    result = ToolResult(
                        response=f"❌ No encontré la secretaría que buscas. Por favor verifica el nombre o consulta la lista completa.",
                        sources=["Google Sheet Horarios Secretarías"],
                        metadata={"match_type": "none", "query_components": intent['components']}
                    )
                    logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
                    return result
            
            # 5. Analizar scores para determinar qué devolver
            best_match = matches[0]
            score, row, match_details = best_match
            
            # Si hay coincidencia excelente (>0.7), devolver solo esa
            if score > 0.7:
                logger.info(f"Coincidencia específica encontrada: {row[0]} (score: {score:.3f})")
                response = self._format_secretaria_info(row, intent)
                result = ToolResult(
                    response=response,
                    sources=["Google Sheet Horarios Secretarías"],
                    metadata={
                        "name": row[0], 
                        "score": score, 
                        "match_type": "specific",
                        "match_details": match_details
                    }
                )
                logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
                return result
            
            # Si hay varias coincidencias buenas (>0.4), devolver las mejores
            good_matches = [(s, r, d) for s, r, d in matches if s > 0.4]
            if 1 <= len(good_matches) <= 5:
                logger.info(f"Devolviendo {len(good_matches)} coincidencias buenas")
                result = self._format_multiple_secretarias(good_matches, intent)
                logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
                return result
            
            # Si el mejor score es bajo (<0.4) o hay demasiadas coincidencias
            if score < 0.4:
                logger.info(f"Score muy bajo ({score:.3f}), buscando alternativas")
                # Si pidió algo específico pero el match es malo, sugerir alternativas
                result = ToolResult(
                    response=f"🔍 No encontré exactamente lo que buscas. Estas son las opciones más cercanas:\n\n" + 
                            self._format_multiple_secretarias(matches[:3], intent).response,
                    sources=["Google Sheet Horarios Secretarías"],
                    metadata={"match_type": "suggestions", "best_score": score}
                )
                logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
                return result
            
            # Si hay demasiadas coincidencias mediocres, mostrar las mejores
            logger.info("Múltiples coincidencias parciales, mostrando las mejores")
            result = self._format_multiple_secretarias(matches[:5], intent)
            logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
            return result
            
        except SecretariaDataError as e:
            logger.error(f"Error de datos: {e}")
            result = ToolResult(
                response="⚠️ Error temporal accediendo a los datos de secretarías. Intenta nuevamente.",
                sources=[],
                metadata={"error": "data_error", "message": str(e)}
            )
            logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Error inesperado en HorariosSecretariasTool: {e}")
            result = ToolResult(
                response="❌ Error interno procesando la consulta. Contacta al administrador.",
                sources=[],
                metadata={"error": "internal_error", "message": str(e)}
            )
            logger.info(f"Consulta completada en {time.time() - start_time:.2f}s")
            return result

    def _format_multiple_secretarias(self, matches: List[Tuple[float, List[str], Dict[str, Any]]], intent: Dict[str, Any]) -> ToolResult:
        """Formatea múltiples secretarías encontradas."""
        results = []
        for score, row, match_details in matches:
            formatted = self._format_secretaria_info(row, intent)
            # Agregar score para transparencia en matches parciales
            if score < 0.7:
                formatted += f"\n  📊 Coincidencia: {score*100:.0f}%"
            results.append(formatted)
        
        response = "🏢 SECRETARÍAS ENCONTRADAS\n\n" + "\n\n".join(results)
        return ToolResult(
            response=response,
            sources=["Google Sheet Horarios Secretarías"],
            metadata={"match_type": "multiple", "count": len(matches), "scores": [m[0] for m in matches]}
        )

    def _format_all_secretarias(self, rows: List[List[str]], intent: Dict[str, Any]) -> ToolResult:
        """Formatea todas las secretarías disponibles."""
        results = []
        
        for row in rows:
            if not row or len(row) < 10:
                continue
            
            # Filtrar por día si se especifica
            if intent['day_requested']:
                day_map = {'L': 1, 'M': 2, 'W': 3, 'J': 4, 'V': 5}
                day_idx = day_map.get(intent['day_requested'])
                if day_idx and len(row) > day_idx:
                    day_flag = "✔" if str(row[day_idx]).strip() else "—"
                    if day_flag != '✔':
                        continue
            
            formatted = self._format_secretaria_info(row, intent)
            results.append(formatted)
        
        if not results:
            return ToolResult(
                response="❌ No se encontraron secretarías que coincidan con tu consulta.",
                sources=["Google Sheet Horarios Secretarías"],
                metadata={"match_type": "none"}
            )
        
        # Limitar a 8 resultados para evitar respuestas muy largas
        preview = "\n\n".join(results[:8])
        if len(results) > 8:
            preview += f"\n\n(+{len(results)-8} más secretarías disponibles)"
        
        return ToolResult(
            response=f"🏢 HORARIOS DE SECRETARÍAS\n\n{preview}",
            sources=["Google Sheet Horarios Secretarías"],
            metadata={"match_type": "all", "total_count": len(results), "shown_count": min(8, len(results))}
        )


