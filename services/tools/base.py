import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from abc import ABC, abstractmethod

# Librerías modernas para fuzzy matching y NLP
from rapidfuzz import fuzz, process
from fuzzywuzzy import fuzz as fw_fuzz
from utils.text_utils import normalize_text as norm

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    score: float
    params: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    response: str
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchDetails:
    """Detalles de matching para debugging y análisis"""
    exact_match: bool = False
    fuzzy_scores: Dict[str, float] = field(default_factory=dict)
    keyword_matches: List[str] = field(default_factory=list)
    component_matches: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class BaseTool(Protocol):
    name: str
    priority: int
    config: Dict[str, Any]

    def configure(self, config: Dict[str, Any]) -> None: ...
    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision: ...
    def accepts(self, score: float) -> bool: ...
    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult: ...


class ModernBaseTool(ABC):
    """
    Clase base moderna para todas las tools con funcionalidades avanzadas:
    - Fuzzy matching con rapidfuzz
    - Caché inteligente
    - Scoring sofisticado
    - Logging detallado
    - Manejo de errores robusto
    """
    
    def __init__(self, name: str, priority: int, default_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.priority = priority
        self.config: Dict[str, Any] = default_config or {
            'thresholds': {'accept': 0.6},
            'triggers': {'keywords': []},
            'fuzzy_matching': {
                'enabled': True,
                'threshold': 0.7,
                'weights': {
                    'ratio': 0.3,
                    'partial': 0.25,
                    'token_sort': 0.25,
                    'token_set': 0.2
                }
            },
            'caching': {
                'enabled': True,
                'ttl_minutes': 30
            }
        }
        self._last_cache_time = 0
        self._cached_data = None
        
    def configure(self, config: Dict[str, Any]) -> None:
        """Configura la tool con nuevos parámetros"""
        if not config:
            return
        self.config.update(config)
        logger.debug(f"Tool {self.name} configurada con: {config}")
    
    @lru_cache(maxsize=128)
    def _normalize_query(self, query: str) -> str:
        """Normaliza una consulta con caché"""
        return norm(query)
    
    def _calculate_fuzzy_score(self, query: str, target: str) -> Tuple[float, MatchDetails]:
        """
        Calcula score de fuzzy matching usando rapidfuzz
        Retorna: (score, match_details)
        """
        start_time = time.time()
        
        query_norm = self._normalize_query(query)
        target_norm = self._normalize_query(target)
        
        match_details = MatchDetails()
        
        # 1. Coincidencia exacta
        if query_norm == target_norm:
            match_details.exact_match = True
            match_details.processing_time = time.time() - start_time
            return 1.0, match_details
        
        # 2. Fuzzy matching con rapidfuzz
        if not self.config.get('fuzzy_matching', {}).get('enabled', True):
            match_details.processing_time = time.time() - start_time
            return 0.0, match_details
        
        weights = self.config.get('fuzzy_matching', {}).get('weights', {
            'ratio': 0.3, 'partial': 0.25, 'token_sort': 0.25, 'token_set': 0.2
        })
        
        ratio_score = fuzz.ratio(query_norm, target_norm) / 100.0
        partial_score = fuzz.partial_ratio(query_norm, target_norm) / 100.0
        token_sort_score = fuzz.token_sort_ratio(query_norm, target_norm) / 100.0
        token_set_score = fuzz.token_set_ratio(query_norm, target_norm) / 100.0
        
        # Combinar scores con pesos
        fuzzy_score = (
            ratio_score * weights['ratio'] +
            partial_score * weights['partial'] +
            token_sort_score * weights['token_sort'] +
            token_set_score * weights['token_set']
        )
        
        match_details.fuzzy_scores = {
            'ratio': ratio_score,
            'partial': partial_score,
            'token_sort': token_sort_score,
            'token_set': token_set_score,
            'combined': fuzzy_score
        }
        
        match_details.processing_time = time.time() - start_time
        return fuzzy_score, match_details
    
    def _advanced_keyword_matching(self, query: str, keywords: List[str]) -> Tuple[float, MatchDetails]:
        """
        Matching avanzado de keywords con fuzzy matching
        Retorna: (score, match_details)
        """
        start_time = time.time()
        query_norm = self._normalize_query(query)
        
        match_details = MatchDetails()
        total_score = 0.0
        matched_keywords = []
        
        for keyword in keywords:
            keyword_norm = self._normalize_query(keyword)
            
            # Coincidencia exacta
            if keyword_norm in query_norm:
                total_score += 1.0
                matched_keywords.append(keyword)
                continue
            
            # Fuzzy matching para keywords
            fuzzy_score, _ = self._calculate_fuzzy_score(query, keyword)
            if fuzzy_score > 0.6:  # Threshold para keywords
                total_score += fuzzy_score * 0.8  # Reducir peso para fuzzy
                matched_keywords.append(f"{keyword} (fuzzy: {fuzzy_score:.2f})")
        
        # Normalizar score
        max_possible = len(keywords)
        final_score = min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0
        
        match_details.keyword_matches = matched_keywords
        match_details.processing_time = time.time() - start_time
        
        return final_score, match_details
    
    def _get_cached_data(self, cache_key: str, fetch_fn, ttl_minutes: Optional[int] = None) -> Any:
        """
        Sistema de caché inteligente con TTL
        """
        if not self.config.get('caching', {}).get('enabled', True):
            return fetch_fn()
        
        ttl = ttl_minutes or self.config.get('caching', {}).get('ttl_minutes', 30)
        current_time = time.time()
        
        # Verificar si el caché es válido
        if (self._cached_data is not None and 
            current_time - self._last_cache_time < ttl * 60):
            logger.debug(f"Usando datos en caché para {cache_key}")
            return self._cached_data
        
        # Actualizar caché
        logger.debug(f"Actualizando caché para {cache_key}")
        self._cached_data = fetch_fn()
        self._last_cache_time = current_time
        return self._cached_data
    
    def _rule_score(self, query: str) -> float:
        """
        Calcula score básico basado en keywords
        Puede ser sobrescrito por subclases para lógica específica
        """
        keywords = self.config.get('triggers', {}).get('keywords', [])
        if not keywords:
            return 0.0
        
        score, _ = self._advanced_keyword_matching(query, keywords)
        return score
    
    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        """
        Determina si la tool puede manejar la consulta
        Puede ser sobrescrito por subclases
        """
        score = self._rule_score(query)
        
        # Boost por contexto previo
        last_query_type = (context or {}).get('last_query_type', '')
        if isinstance(last_query_type, str) and self.name in last_query_type:
            score = max(score, 0.7)
        
        return Decision(
            score=score, 
            params={}, 
            reasons=[f"{self.name}_rule_score"]
        )
    
    def accepts(self, score: float) -> bool:
        """Determina si acepta el score dado"""
        threshold = float(self.config.get('thresholds', {}).get('accept', 0.6))
        return score >= threshold
    
    @abstractmethod
    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        """Ejecuta la funcionalidad específica de la tool"""
        pass


class SheetsBaseTool(ModernBaseTool):
    """
    Clase base especializada para tools que usan Google Sheets
    Incluye funcionalidades específicas para manejo de datos de sheets
    """
    
    def __init__(self, name: str, priority: int, sheets_service, default_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, priority, default_config)
        self.sheets_service = sheets_service
        
        # Configuración específica para sheets
        sheets_config = {
            'spreadsheet_id': None,
            'sheet_name': 'Hoja 1',
            'ranges': {'default': 'A:Z'},
            'data_processing': {
                'skip_header_rows': 1,
                'normalize_columns': True,
                'handle_merged_cells': True
            }
        }
        self.config.update(sheets_config)
        if default_config:
            self.config.update(default_config)
    
    def _fetch_sheet_data(self, spreadsheet_id: Optional[str] = None, 
                         sheet_name: Optional[str] = None, 
                         range_name: Optional[str] = None) -> List[List[Any]]:
        """
        Obtiene datos del sheet con caché inteligente
        """
        sid = spreadsheet_id or self.config.get('spreadsheet_id')
        sheet = sheet_name or self.config.get('sheet_name', 'Hoja 1')
        rng = range_name or self.config.get('ranges', {}).get('default', 'A:Z')
        
        if not sid:
            logger.warning(f"Tool {self.name}: No hay spreadsheet_id configurado")
            return []
        
        cache_key = f"sheet_{sid}_{sheet}_{rng}"
        
        def fetch_data():
            try:
                a1_range = f"'{sheet}'!{rng}"
                values = self.sheets_service.get_sheet_values(sid, a1_range)
                logger.debug(f"Tool {self.name}: Obtenidos {len(values) if values else 0} filas del sheet")
                return values or []
            except Exception as e:
                logger.error(f"Tool {self.name}: Error obteniendo datos del sheet: {e}")
                return []
        
        return self._get_cached_data(cache_key, fetch_data)
    
    def _process_sheet_rows(self, rows: List[List[Any]]) -> List[List[Any]]:
        """
        Procesa filas del sheet aplicando normalizaciones
        """
        if not rows:
            return []
        
        skip_rows = self.config.get('data_processing', {}).get('skip_header_rows', 1)
        processed_rows = rows[skip_rows:] if len(rows) > skip_rows else []
        
        # Manejar celdas combinadas (propagar valores hacia abajo)
        if self.config.get('data_processing', {}).get('handle_merged_cells', True):
            processed_rows = self._handle_merged_cells(processed_rows)
        
        return processed_rows
    
    def _handle_merged_cells(self, rows: List[List[Any]]) -> List[List[Any]]:
        """
        Maneja celdas combinadas propagando valores hacia abajo
        """
        if not rows:
            return []
        
        processed = []
        current_values = {}
        
        for row in rows:
            if not row:
                processed.append(row)
                continue
            
            # Crear copia de la fila
            new_row = list(row)
            
            # Propagar valores de celdas no vacías
            for i, cell in enumerate(new_row):
                if cell and str(cell).strip():
                    current_values[i] = str(cell).strip()
                elif i in current_values:
                    new_row[i] = current_values[i]
            
            processed.append(new_row)
        
        return processed
    
    def _find_best_matches(self, query: str, data: List[List[Any]], 
                          search_columns: List[int], 
                          threshold: float = 0.6) -> List[Tuple[int, float, MatchDetails]]:
        """
        Encuentra las mejores coincidencias en los datos del sheet
        """
        query_norm = self._normalize_query(query)
        matches = []
        
        for row_idx, row in enumerate(data):
            if not row:
                continue
            
            best_score = 0.0
            best_details = MatchDetails()
            
            # Buscar en columnas específicas
            for col_idx in search_columns:
                if col_idx < len(row) and row[col_idx]:
                    cell_value = str(row[col_idx]).strip()
                    if not cell_value:
                        continue
                    
                    score, details = self._calculate_fuzzy_score(query_norm, cell_value)
                    if score > best_score:
                        best_score = score
                        best_details = details
            
            if best_score >= threshold:
                matches.append((row_idx, best_score, best_details))
        
        # Ordenar por score descendente
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


class ConversationalBaseTool(ModernBaseTool):
    """
    Clase base para tools conversacionales con NLP avanzado
    """
    
    def __init__(self, name: str, priority: int, model=None, default_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, priority, default_config)
        self.model = model
        
        # Configuración específica para conversacional
        conversational_config = {
            'thresholds': {'accept': 0.5},
            'intent_detection': {
                'enabled': True,
                'confidence_threshold': 0.7
            },
            'context_awareness': {
                'enabled': True,
                'context_boost': 0.3
            }
        }
        self.config.update(conversational_config)
        if default_config:
            self.config.update(default_config)
    
    def _detect_intent(self, query: str, context: Dict[str, Any]) -> Tuple[str, float]:
        """
        Detecta la intención de la consulta usando NLP
        """
        # Implementación básica - puede ser extendida con modelos más avanzados
        query_norm = self._normalize_query(query)
        
        # Detectar intents básicos por longitud y patrones
        if len(query.split()) <= 4:
            return "greeting", 0.8
        
        if any(word in query_norm for word in ["gracias", "thank", "thanks"]):
            return "gratitude", 0.9
        
        if any(word in query_norm for word in ["ayuda", "help", "que puedes", "que podes"]):
            return "help", 0.8
        
        return "general", 0.5
    
    def can_handle(self, query: str, context: Dict[str, Any]) -> Decision:
        """
        Determina si puede manejar consultas conversacionales
        """
        score = self._rule_score(query)
        
        # Boost por longitud de consulta (consultas cortas son más conversacionales)
        if len(query.split()) <= 4:
            score = max(score, 0.6)
        
        # Boost por contexto conversacional
        if self.config.get('context_awareness', {}).get('enabled', True):
            last_query_type = (context or {}).get('last_query_type', '')
            if 'conversational' in last_query_type:
                boost = self.config.get('context_awareness', {}).get('context_boost', 0.3)
                score = max(score, score + boost)
        
        return Decision(
            score=score,
            params={},
            reasons=[f"{self.name}_conversational_score"]
        )


