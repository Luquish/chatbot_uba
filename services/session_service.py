"""
Servicio para manejar sesiones temporales de conversación.
Mantiene contexto conversacional en memoria con TTL automático.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from threading import Lock
from config.settings import config
from .relative_query_processors import RelativeQueryManager
from .llm_context_resolver import HybridContextResolver

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """Datos de sesión de usuario."""
    user_id: str
    last_query: str = ""
    last_query_type: str = ""  # cursos, calendario, faq, materia, docente, tramite, biblioteca, etc.
    last_calendar_intent: str = ""  # eventos_generales, examenes, inscripciones, etc.
    last_month_requested: str = ""  # Para cursos
    last_time_reference: str = ""  # esta semana, este mes, etc.
    last_subject_requested: str = ""  # Para materias específicas
    last_teacher_requested: str = ""  # Para docentes específicos
    last_procedure_requested: str = ""  # Para trámites específicos
    last_resource_requested: str = ""  # Para recursos específicos (biblioteca, etc.)
    last_department_requested: str = ""  # Para departamentos específicos
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

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600, enable_background_sweeper: bool = True, relative_query_manager: Optional[RelativeQueryManager] = None):
        """
        Inicializa el servicio de sesiones.
        
        Args:
            max_sessions: Número máximo de sesiones simultáneas
            ttl_seconds: Tiempo de vida de las sesiones en segundos (por defecto 1 hora)
        """
        self.sessions: Dict[str, UserSession] = {}
        self.max_sessions = max_sessions
        # Usar TTL desde settings si está disponible; por defecto 30 minutos
        config_ttl = None
        try:
            config_ttl = config.session.ttl_seconds
        except Exception:
            config_ttl = None
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else (config_ttl or 1800)
        self._lock = Lock()
        # Configurar barrido en segundo plano
        self._sweeper_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sweeper_interval = getattr(getattr(config, 'session', None), 'sweeper_interval_seconds', 60)
        # Gestor de consultas relativas (híbrido: patrones + LLM)
        self.relative_query_manager = relative_query_manager or RelativeQueryManager()
        self.llm_context_resolver = HybridContextResolver()
        if enable_background_sweeper:
            self._start_background_sweeper()
        logger.info(f"SessionService inicializado. Max sesiones: {max_sessions}, TTL: {self.ttl_seconds}s, Sweeper: {enable_background_sweeper}")

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
                              time_reference: str = None, subject_requested: str = None,
                              teacher_requested: str = None, procedure_requested: str = None,
                              resource_requested: str = None, department_requested: str = None,
                              user_name: Optional[str] = None, **kwargs):
        """
        Actualiza el contexto de la sesión del usuario.
        
        Args:
            user_id: ID del usuario
            query: Consulta realizada
            query_type: Tipo de consulta (cursos, calendario, materia, docente, tramite, biblioteca, etc.)
            month_requested: Mes solicitado en la consulta (para cursos)
            calendar_intent: Tipo de intent de calendario (eventos_generales, examenes, etc.)
            time_reference: Referencia temporal (esta semana, este mes, etc.)
            subject_requested: Materia específica mencionada (para materias)
            teacher_requested: Docente específico mencionado (para docentes)
            procedure_requested: Trámite específico mencionado (para trámites)
            resource_requested: Recurso específico mencionado (para biblioteca, etc.)
            department_requested: Departamento específico mencionado
            user_name: Nombre del usuario si está disponible
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
        
        if subject_requested:
            session.last_subject_requested = subject_requested
        
        if teacher_requested:
            session.last_teacher_requested = teacher_requested
        
        if procedure_requested:
            session.last_procedure_requested = procedure_requested
        
        if resource_requested:
            session.last_resource_requested = resource_requested
        
        if department_requested:
            session.last_department_requested = department_requested
        
        if user_name:
            session.context_data['user_name'] = user_name
        
        # Agregar datos adicionales al contexto
        for key, value in kwargs.items():
            session.context_data[key] = value
        
        logger.info(f"Contexto actualizado para {user_id}: tipo={query_type}, mes={month_requested}, calendar_intent={calendar_intent}, time_ref={time_reference}, subject={subject_requested}, teacher={teacher_requested}, procedure={procedure_requested}, resource={resource_requested}")

    def get_context_for_relative_query(self, user_id: str, query: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Analiza una consulta relativa y devuelve el contexto necesario.
        Usa enfoque híbrido: patrones rápidos + LLM para flexibilidad.
        
        Args:
            user_id: ID del usuario
            query: Consulta actual
            use_llm: Si usar LLM para casos complejos (default: True)
            
        Returns:
            Dict: Contexto interpretado para la consulta
        """
        session = self.get_session(user_id)
        
        if use_llm:
            # Enfoque híbrido: primero patrones rápidos, luego LLM si es necesario
            session_context = {
                'last_query': session.last_query,
                'last_query_type': session.last_query_type,
                'last_month_requested': session.last_month_requested,
                'last_time_reference': session.last_time_reference,
                'last_calendar_intent': session.last_calendar_intent,
                'last_subject_requested': session.last_subject_requested,
                'last_teacher_requested': session.last_teacher_requested,
                'last_procedure_requested': session.last_procedure_requested,
                'last_resource_requested': session.last_resource_requested,
                'last_department_requested': session.last_department_requested
            }
            
            try:
                llm_result = self.llm_context_resolver.resolve_relative_query(query, session_context)
                
                if llm_result.is_relative:
                    # Convertir resultado LLM al formato esperado
                    context = {
                        "is_relative": True,
                        "resolved_month": llm_result.resolved_context if llm_result.context_type == "month" else None,
                        "resolved_time_reference": llm_result.resolved_context if llm_result.context_type in ["week", "time_period"] else None,
                        "query_type": session.last_query_type,
                        "calendar_intent": session.last_calendar_intent,
                        "relative_offset": llm_result.offset,
                        "explanation": llm_result.explanation,
                        "confidence": llm_result.confidence,
                        "method": "llm_hybrid"
                    }
                    
                    logger.info(f"LLM detectó consulta relativa para {user_id}: {llm_result.explanation} (confianza: {llm_result.confidence:.2f})")
                    return context
                else:
                    # No es relativa según LLM
                    return {
                        "is_relative": False,
                        "resolved_month": None,
                        "resolved_time_reference": None,
                        "query_type": session.last_query_type,
                        "calendar_intent": session.last_calendar_intent,
                        "relative_offset": 0,
                        "explanation": llm_result.explanation,
                        "method": "llm_hybrid"
                    }
                    
            except Exception as e:
                logger.warning(f"Error con LLM para usuario {user_id}, usando patrones como fallback: {e}")
                # Fallback a patrones tradicionales
                result = self.relative_query_manager.get_context_for_relative_query(session, query, user_id)
                result["method"] = "pattern_fallback"
                return result
        else:
            # Usar solo patrones (incluyendo los nuevos patrones híbridos)
            session_context = {
                'last_query': session.last_query,
                'last_query_type': session.last_query_type,
                'last_month_requested': session.last_month_requested,
                'last_time_reference': session.last_time_reference,
                'last_calendar_intent': session.last_calendar_intent,
                'last_subject_requested': session.last_subject_requested,
                'last_teacher_requested': session.last_teacher_requested,
                'last_procedure_requested': session.last_procedure_requested,
                'last_resource_requested': session.last_resource_requested,
                'last_department_requested': session.last_department_requested
            }
            
            # Usar HybridContextResolver pero sin LLM
            hybrid_result = self.llm_context_resolver.resolve_relative_query(query, session_context)
            
            if hybrid_result.is_relative:
                return {
                    "is_relative": True,
                    "resolved_month": hybrid_result.resolved_context if hybrid_result.context_type == "month" else None,
                    "resolved_time_reference": hybrid_result.resolved_context if hybrid_result.context_type in ["week", "time_period"] else None,
                    "resolved_subject": hybrid_result.resolved_context if hybrid_result.context_type == "subject" else None,
                    "resolved_teacher": hybrid_result.resolved_context if hybrid_result.context_type == "teacher" else None,
                    "resolved_procedure": hybrid_result.resolved_context if hybrid_result.context_type == "procedure" else None,
                    "resolved_library": hybrid_result.resolved_context if hybrid_result.context_type == "library" else None,
                    "query_type": session.last_query_type,
                    "calendar_intent": session.last_calendar_intent,
                    "relative_offset": hybrid_result.offset,
                    "explanation": hybrid_result.explanation,
                    "method": "pattern_only"
                }
            else:
                return {
                    "is_relative": False,
                    "resolved_month": None,
                    "resolved_time_reference": None,
                    "resolved_subject": None,
                    "resolved_teacher": None,
                    "resolved_procedure": None,
                    "resolved_library": None,
                    "query_type": session.last_query_type,
                    "calendar_intent": session.last_calendar_intent,
                    "relative_offset": 0,
                    "explanation": hybrid_result.explanation,
                    "method": "pattern_only"
                }

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

    def _start_background_sweeper(self):
        """Inicia un hilo en segundo plano que limpia sesiones expiradas periódicamente."""
        if self._sweeper_thread and self._sweeper_thread.is_alive():
            return

        def _sweeper_loop():
            logger.info("SessionService sweeper iniciado")
            while not self._stop_event.is_set():
                try:
                    with self._lock:
                        self._cleanup_expired_sessions()
                except Exception as e:
                    logger.warning(f"Error en sweeper de sesiones: {e}")
                # Espera con despertar anticipado si se detiene
                self._stop_event.wait(self._sweeper_interval)
            logger.info("SessionService sweeper detenido")

        self._sweeper_thread = threading.Thread(target=_sweeper_loop, name="SessionSweeper", daemon=True)
        self._sweeper_thread.start()

    def stop(self):
        """Detiene el sweeper en segundo plano si está activo."""
        self._stop_event.set()
        if self._sweeper_thread and self._sweeper_thread.is_alive():
            self._sweeper_thread.join(timeout=2)

    def get_session_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas del servicio de sesiones."""
        with self._lock:
            self._cleanup_expired_sessions()
            return {
                "active_sessions": len(self.sessions),
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds
            }


class SessionServiceSingleton:
    """Singleton para el servicio de sesiones con soporte para testing."""
    _instance: Optional[SessionService] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, **kwargs) -> SessionService:
        """Obtiene o crea la instancia singleton del SessionService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Usar TTL desde settings si está disponible
                    try:
                        default_ttl = config.session.ttl_seconds if hasattr(config, 'session') and getattr(config.session, 'ttl_seconds', None) else 1800
                    except Exception:
                        default_ttl = 1800
                    
                    # Permitir override de parámetros para testing
                    default_kwargs = {
                        'max_sessions': 1000,
                        'ttl_seconds': default_ttl,
                        'enable_background_sweeper': True
                    }
                    default_kwargs.update(kwargs)
                    
                    cls._instance = SessionService(**default_kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reinicia la instancia singleton (útil para testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.stop()  # Detener sweeper si está activo
            cls._instance = None
    
    @classmethod
    def set_instance(cls, instance: SessionService):
        """Establece una instancia específica (útil para testing con mocks)."""
        with cls._lock:
            if cls._instance:
                cls._instance.stop()
            cls._instance = instance


# Función de conveniencia para obtener la instancia global
def get_session_service(**kwargs) -> SessionService:
    """Obtiene la instancia global del SessionService."""
    return SessionServiceSingleton.get_instance(**kwargs)


# Mantener compatibilidad hacia atrás
session_service = get_session_service()
