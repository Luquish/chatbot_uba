"""
Servicio para manejar sesiones temporales de conversación.
Mantiene contexto conversacional en memoria con TTL automático.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Datos de sesión de usuario."""
    user_id: str
    last_query: str = ""
    last_query_type: str = ""  # cursos, calendario, faq, etc.
    last_calendar_intent: str = ""  # eventos_generales, examenes, inscripciones, etc.
    last_month_requested: str = ""  # Para cursos
    last_time_reference: str = ""  # esta semana, este mes, etc.
    last_activity: float = 0
    context_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.context_data is None:
            self.context_data = {}
        self.last_activity = time.time()

    def update_activity(self):
        """Actualiza el timestamp de última actividad."""
        self.last_activity = time.time()

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Verifica si la sesión ha expirado (por defecto 1 hora)."""
        return time.time() - self.last_activity > ttl_seconds


class SessionService:
    """Servicio para manejar sesiones temporales en memoria."""

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600):
        """
        Inicializa el servicio de sesiones.
        
        Args:
            max_sessions: Número máximo de sesiones simultáneas
            ttl_seconds: Tiempo de vida de las sesiones en segundos (por defecto 1 hora)
        """
        self.sessions: Dict[str, UserSession] = {}
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._lock = Lock()
        logger.info(f"SessionService inicializado. Max sesiones: {max_sessions}, TTL: {ttl_seconds}s")

    def get_session(self, user_id: str) -> UserSession:
        """
        Obtiene o crea una sesión para el usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            UserSession: Sesión del usuario
        """
        with self._lock:
            # Limpiar sesiones expiradas antes de crear nuevas
            self._cleanup_expired_sessions()
            
            if user_id not in self.sessions:
                # Verificar límite de sesiones
                if len(self.sessions) >= self.max_sessions:
                    # Eliminar la sesión más antigua
                    oldest_user = min(self.sessions.keys(), 
                                     key=lambda k: self.sessions[k].last_activity)
                    del self.sessions[oldest_user]
                    logger.info(f"Sesión eliminada por límite: {oldest_user}")
                
                # Crear nueva sesión
                self.sessions[user_id] = UserSession(user_id=user_id)
                logger.info(f"Nueva sesión creada para usuario: {user_id}")
            
            # Actualizar actividad
            self.sessions[user_id].update_activity()
            return self.sessions[user_id]

    def update_session_context(self, user_id: str, query: str, query_type: str, 
                              month_requested: str = None, calendar_intent: str = None,
                              time_reference: str = None, **kwargs):
        """
        Actualiza el contexto de la sesión del usuario.
        
        Args:
            user_id: ID del usuario
            query: Consulta realizada
            query_type: Tipo de consulta (cursos, calendario, etc.)
            month_requested: Mes solicitado en la consulta (para cursos)
            calendar_intent: Tipo de intent de calendario (eventos_generales, examenes, etc.)
            time_reference: Referencia temporal (esta semana, este mes, etc.)
            **kwargs: Datos adicionales de contexto
        """
        session = self.get_session(user_id)
        
        session.last_query = query
        session.last_query_type = query_type
        
        if month_requested:
            session.last_month_requested = month_requested
        
        if calendar_intent:
            session.last_calendar_intent = calendar_intent
            
        if time_reference:
            session.last_time_reference = time_reference
        
        # Agregar datos adicionales al contexto
        for key, value in kwargs.items():
            session.context_data[key] = value
        
        logger.info(f"Contexto actualizado para {user_id}: tipo={query_type}, mes={month_requested}, calendar_intent={calendar_intent}, time_ref={time_reference}")

    def get_context_for_relative_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        Analiza una consulta relativa y devuelve el contexto necesario para tanto cursos como calendario.
        
        Args:
            user_id: ID del usuario
            query: Consulta actual
            
        Returns:
            Dict: Contexto interpretado para la consulta
        """
        session = self.get_session(user_id)
        query_lower = query.lower()
        
        context = {
            "is_relative": False,
            "resolved_month": None,
            "resolved_time_reference": None,
            "query_type": session.last_query_type,
            "calendar_intent": session.last_calendar_intent,
            "explanation": ""
        }
        
        # Patrones para consultas relativas de MESES (cursos)
        month_relative_patterns = {
            "y en dos meses": 2,
            "en dos meses": 2,
            "y en tres meses": 3,
            "en tres meses": 3,
            "el mes pasado": -1,
            "mes anterior": -1,
            "y el anterior": -1,
            "y el siguiente": 1,
            "y el que viene": 1,  # NUEVO
            "y el que sigue": 1,  # NUEVO
            "el que viene": 1,    # NUEVO
            "el que sigue": 1,    # NUEVO
            "próximo mes": 1,
            "siguiente mes": 1
        }
        
        # Patrones para consultas relativas de TIEMPO (calendario)
        time_relative_patterns = {
            # Semanas
            "la semana anterior": {"type": "week", "offset": -1},
            "semana anterior": {"type": "week", "offset": -1},
            "la anterior": {"type": "week", "offset": -1},  # contexto de semana
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
            "y el que viene": {"type": "context", "offset": 1},  # NUEVO
            "y el que sigue": {"type": "context", "offset": 1},  # NUEVO
            "el que viene": {"type": "context", "offset": 1},    # NUEVO
            "el que sigue": {"type": "context", "offset": 1}     # NUEVO
        }
        
        # **PRIORIDAD 1: Detectar consultas relativas de CURSOS (meses)**
        if session.last_query_type == "cursos" and session.last_month_requested:
            for pattern, month_offset in month_relative_patterns.items():
                if pattern in query_lower:
                    context["is_relative"] = True
                    try:
                        months_list = ['ENERO', 'FEBRERO', 'MARZO', 'ABRIL', 'MAYO', 'JUNIO',
                                     'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE', 'DICIEMBRE']
                        
                        current_month_idx = months_list.index(session.last_month_requested)
                        new_month_idx = (current_month_idx + month_offset) % 12
                        context["resolved_month"] = months_list[new_month_idx]
                        context["explanation"] = f"Interpretando '{pattern}' como {context['resolved_month']} (basado en {session.last_month_requested})"
                        
                        logger.info(f"Consulta relativa de CURSOS detectada para {user_id}: {session.last_month_requested} + {month_offset} = {context['resolved_month']}")
                        return context
                        
                    except (ValueError, IndexError):
                        logger.warning(f"Error al calcular mes relativo para {user_id}")
                    break
        
        # **PRIORIDAD 2: Detectar consultas relativas de CALENDARIO**
        if session.last_query_type.startswith("calendario") and session.last_time_reference:
            for pattern, time_config in time_relative_patterns.items():
                if pattern in query_lower:
                    context["is_relative"] = True
                    
                    # Determinar el tipo de referencia temporal basado en contexto
                    if time_config["type"] == "context":
                        # Usar el contexto de la sesión anterior
                        if "semana" in session.last_time_reference.lower():
                            time_config["type"] = "week"
                        elif "mes" in session.last_time_reference.lower():
                            time_config["type"] = "month"
                    
                    # Calcular nueva referencia temporal
                    if time_config["type"] == "week":
                        if time_config["offset"] == -1:
                            context["resolved_time_reference"] = "la semana pasada"
                        elif time_config["offset"] == 1:
                            context["resolved_time_reference"] = "la próxima semana"
                        else:
                            context["resolved_time_reference"] = f"en {abs(time_config['offset'])} semanas"
                    
                    elif time_config["type"] == "month":
                        if time_config["offset"] == -1:
                            context["resolved_time_reference"] = "el mes pasado"
                        elif time_config["offset"] == 1:
                            context["resolved_time_reference"] = "el próximo mes"
                        else:
                            context["resolved_time_reference"] = f"en {abs(time_config['offset'])} meses"
                    
                    context["explanation"] = f"Interpretando '{pattern}' como '{context['resolved_time_reference']}' para {session.last_calendar_intent}"
                    
                    logger.info(f"Consulta relativa de CALENDARIO detectada para {user_id}: {session.last_time_reference} + {time_config['offset']} = {context['resolved_time_reference']}")
                    return context
        
        return context

    def _cleanup_expired_sessions(self):
        """Limpia sesiones expiradas."""
        current_time = time.time()
        expired_users = [
            user_id for user_id, session in self.sessions.items()
            if session.is_expired(self.ttl_seconds)
        ]
        
        for user_id in expired_users:
            del self.sessions[user_id]
        
        if expired_users:
            logger.info(f"Sesiones expiradas eliminadas: {len(expired_users)}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas del servicio de sesiones."""
        with self._lock:
            self._cleanup_expired_sessions()
            return {
                "active_sessions": len(self.sessions),
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds
            }


# Instancia global del servicio de sesiones
session_service = SessionService(max_sessions=1000, ttl_seconds=3600)  # 1 hora TTL
