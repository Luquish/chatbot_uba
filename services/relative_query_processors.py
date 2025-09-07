"""
Procesadores especializados para consultas relativas.
Separa la lógica compleja en clases dedicadas.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RelativeContext:
    """Contexto resuelto para una consulta relativa."""
    is_relative: bool = False
    resolved_month: Optional[str] = None
    resolved_time_reference: Optional[str] = None
    query_type: str = ""
    calendar_intent: str = ""
    relative_offset: int = 0
    explanation: str = ""


class CourseRelativeProcessor:
    """Procesador de consultas relativas para cursos (meses)."""
    
    MONTHS = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO',
              'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
    
    MONTH_PATTERNS = {
        "y en dos meses": 2,
        "en dos meses": 2,
        "y en tres meses": 3,
        "en tres meses": 3,
        "el mes pasado": -1,
        "mes anterior": -1,
        "y el anterior": -1,
        "y el siguiente": 1,
        "y el que viene": 1,
        "y el que sigue": 1,
        "el que viene": 1,
        "el que sigue": 1,
        "próximo mes": 1,
        "siguiente mes": 1
    }
    
    def can_process(self, session) -> bool:
        """Verifica si puede procesar la consulta basada en la sesión."""
        return session.last_query_type == "cursos" and session.last_month_requested
    
    def process(self, session, query: str, user_id: str) -> Optional[RelativeContext]:
        """Procesa consulta relativa de cursos."""
        query_lower = query.lower()
        
        for pattern, month_offset in self.MONTH_PATTERNS.items():
            if pattern in query_lower:
                try:
                    current_month_idx = self.MONTHS.index(session.last_month_requested)
                    new_month_idx = (current_month_idx + month_offset) % 12
                    resolved_month = self.MONTHS[new_month_idx]
                    
                    context = RelativeContext(
                        is_relative=True,
                        resolved_month=resolved_month,
                        query_type=session.last_query_type,
                        explanation=f"Interpretando '{pattern}' como {resolved_month} (basado en {session.last_month_requested})"
                    )
                    
                    logger.info(f"Consulta relativa de CURSOS detectada para {user_id}: {session.last_month_requested} + {month_offset} = {resolved_month}")
                    return context
                    
                except (ValueError, IndexError):
                    logger.warning(f"Error al calcular mes relativo para {user_id}")
                break
        
        return None


class CalendarRelativeProcessor:
    """Procesador de consultas relativas para calendario (tiempo)."""
    
    TIME_PATTERNS = {
        # Semanas
        "la semana anterior": {"type": "week", "offset": -1},
        "semana anterior": {"type": "week", "offset": -1},
        "la anterior": {"type": "week", "offset": -1},
        "la semana pasada": {"type": "week", "offset": -1},
        "semana pasada": {"type": "week", "offset": -1},
        "la próxima semana": {"type": "week", "offset": 1},
        "próxima semana": {"type": "week", "offset": 1},
        "la siguiente semana": {"type": "week", "offset": 1},
        "siguiente semana": {"type": "week", "offset": 1},
        
        # Meses (para calendario)
        "el mes anterior": {"type": "month", "offset": -1},
        "mes anterior": {"type": "month", "offset": -1},
        "el próximo mes": {"type": "month", "offset": 1},
        "próximo mes": {"type": "month", "offset": 1},
        "el siguiente mes": {"type": "month", "offset": 1},
        "siguiente mes": {"type": "month", "offset": 1},
        
        # Referencias generales que dependen del contexto
        "y el anterior": {"type": "context", "offset": -1},
        "y el siguiente": {"type": "context", "offset": 1},
        "y la anterior": {"type": "context", "offset": -1},
        "y la siguiente": {"type": "context", "offset": 1},
        "y el que viene": {"type": "context", "offset": 1},
        "y el que sigue": {"type": "context", "offset": 1},
        "y la que viene": {"type": "context", "offset": 1},
        "y la que sigue": {"type": "context", "offset": 1},
        "el que viene": {"type": "context", "offset": 1},
        "el que sigue": {"type": "context", "offset": 1}
    }
    
    def can_process(self, session) -> bool:
        """Verifica si puede procesar la consulta basada en la sesión."""
        return session.last_query_type.startswith("calendario") and bool(session.last_time_reference)
    
    def process(self, session, query: str, user_id: str) -> Optional[RelativeContext]:
        """Procesa consulta relativa de calendario."""
        query_lower = query.lower()
        
        for pattern, time_config in self.TIME_PATTERNS.items():
            if pattern in query_lower:
                # Determinar el tipo de referencia temporal basado en contexto
                time_type = time_config["type"]
                if time_type == "context":
                    if "semana" in session.last_time_reference.lower():
                        time_type = "week"
                    elif "mes" in session.last_time_reference.lower():
                        time_type = "month"
                
                resolved_time_ref = self._resolve_time_reference(time_type, time_config["offset"])
                
                context = RelativeContext(
                    is_relative=True,
                    resolved_time_reference=resolved_time_ref,
                    query_type=session.last_query_type,
                    calendar_intent=session.last_calendar_intent,
                    relative_offset=time_config["offset"],
                    explanation=f"Interpretando '{pattern}' como '{resolved_time_ref}' para {session.last_calendar_intent}"
                )
                
                logger.info(f"Consulta relativa de CALENDARIO detectada para {user_id}: {session.last_time_reference} + {time_config['offset']} = {resolved_time_ref}")
                return context
        
        return None
    
    def _resolve_time_reference(self, time_type: str, offset: int) -> str:
        """Resuelve la referencia temporal basada en tipo y offset."""
        if time_type == "week":
            if offset == -1:
                return "la semana pasada"
            elif offset == 1:
                return "la próxima semana"
            else:
                return f"en {abs(offset)} semanas"
        
        elif time_type == "month":
            if offset == -1:
                return "el mes pasado"
            elif offset == 1:
                return "el próximo mes"
            else:
                return f"en {abs(offset)} meses"
        
        return f"offset {offset}"


class RelativeQueryManager:
    """Gestor principal para consultas relativas."""
    
    def __init__(self):
        self.processors = [
            CourseRelativeProcessor(),
            CalendarRelativeProcessor()
        ]
    
    def get_context_for_relative_query(self, session, query: str, user_id: str) -> Dict[str, Any]:
        """
        Analiza una consulta relativa y devuelve el contexto necesario.
        
        Args:
            session: Sesión del usuario
            query: Consulta actual
            user_id: ID del usuario
            
        Returns:
            Dict: Contexto interpretado para la consulta
        """
        # Contexto base
        base_context = RelativeContext(
            query_type=session.last_query_type,
            calendar_intent=session.last_calendar_intent
        )
        
        # Intentar procesar con cada procesador en orden de prioridad
        for processor in self.processors:
            if processor.can_process(session):
                context = processor.process(session, query, user_id)
                if context:
                    return context.__dict__
        
        # Si no se encontró patrón relativo, devolver contexto base
        return base_context.__dict__