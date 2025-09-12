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


# Excepciones específicas para el manejo de sesiones
class SessionError(Exception):
    """Excepción base para errores de sesión."""
    pass


class SessionNotFoundError(SessionError):
    """Se lanza cuando no se encuentra una sesión solicitada."""
    pass


class SessionExpiredError(SessionError):
    """Se lanza cuando se intenta usar una sesión expirada."""
    pass


class InvalidSessionDataError(SessionError):
    """Se lanza cuando los datos de sesión son inválidos."""
    pass

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

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = None, enable_background_sweeper: bool = True, relative_query_manager: Optional[RelativeQueryManager] = None):
        """
        Inicializa el servicio de sesiones.
        
        Args:
            max_sessions: Número máximo de sesiones simultáneas
            ttl_seconds: Tiempo de vida de las sesiones en segundos (None para usar config)
            enable_background_sweeper: Si habilitar el limpiador automático
            relative_query_manager: Gestor de consultas relativas (opcional)
        """
        self.sessions: Dict[str, UserSession] = {}
        self.max_sessions = max_sessions
        
        # Configuración unificada de TTL: usar config.session.ttl_seconds como fuente única de verdad
        if ttl_seconds is not None:
            self.ttl_seconds = ttl_seconds
        else:
            try:
                self.ttl_seconds = config.session.ttl_seconds
            except Exception:
                # Fallback consistente: 30 minutos (1800 segundos)
                self.ttl_seconds = 1800
        
        self._lock = Lock()
        
        # Configurar barrido en segundo plano
        self._sweeper_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        try:
            self._sweeper_interval = config.session.sweeper_interval_seconds
        except Exception:
            self._sweeper_interval = 60  # 1 minuto por defecto
        
        # Gestor de consultas relativas (híbrido: patrones + LLM)
        self.relative_query_manager = relative_query_manager or RelativeQueryManager()
        self.llm_context_resolver = HybridContextResolver()
        
        if enable_background_sweeper:
            self._start_background_sweeper()
            
        logger.info(f"SessionService inicializado. Max sesiones: {max_sessions}, TTL: {self.ttl_seconds}s, Sweeper: {enable_background_sweeper}, Intervalo: {self._sweeper_interval}s")

    def get_session(self, user_id: str) -> UserSession:
        """
        Obtiene o crea una sesión para el usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            UserSession: Sesión del usuario
            
        Raises:
            ValueError: Si user_id es inválido
        """
        # Validar user_id
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise InvalidSessionDataError("user_id debe ser una cadena no vacía")
        
        user_id = user_id.strip()
        
        with self._lock:
            # PRIMERO: Verificar si existe una sesión válida
            if user_id in self.sessions:
                session = self.sessions[user_id]
                # Si la sesión no ha expirado, actualizarla y devolverla
                if not session.is_expired(self.ttl_seconds):
                    session.update_activity()
                    logger.debug(f"Sesión existente actualizada para usuario: {user_id}")
                    return session
                else:
                    # Si expiró, eliminarla
                    session_age = time.time() - session.last_activity
                    del self.sessions[user_id]
                    logger.info(f"Sesión expirada eliminada para usuario: {user_id} (edad: {session_age:.1f}s)")
            
            # SEGUNDO: Limpiar otras sesiones expiradas
            self._cleanup_expired_sessions()
            
            # TERCERO: Verificar límite de sesiones antes de crear nueva
            if len(self.sessions) >= self.max_sessions:
                # Eliminar la sesión más antigua
                oldest_user = min(self.sessions.keys(), 
                                 key=lambda k: self.sessions[k].last_activity)
                del self.sessions[oldest_user]
                logger.info(f"Sesión eliminada por límite: {oldest_user}")
            
            # CUARTO: Crear nueva sesión
            self.sessions[user_id] = UserSession(user_id=user_id)
            logger.info(f"Nueva sesión creada para usuario: {user_id}")
            
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
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validar parámetros requeridos
        if not query or not isinstance(query, str):
            raise InvalidSessionDataError("query debe ser una cadena no vacía")
        
        if not query_type or not isinstance(query_type, str):
            raise InvalidSessionDataError("query_type debe ser una cadena no vacía")
        
        session = self.get_session(user_id)
        
        # Sanitizar y asignar valores
        session.last_query = query.strip()
        session.last_query_type = query_type.strip()
        
        if month_requested:
            session.last_month_requested = month_requested.strip()
        
        if calendar_intent:
            session.last_calendar_intent = calendar_intent.strip()
            
        if time_reference:
            session.last_time_reference = time_reference.strip()
        
        if subject_requested:
            session.last_subject_requested = subject_requested.strip()
        
        if teacher_requested:
            session.last_teacher_requested = teacher_requested.strip()
        
        if procedure_requested:
            session.last_procedure_requested = procedure_requested.strip()
        
        if resource_requested:
            session.last_resource_requested = resource_requested.strip()
        
        if department_requested:
            session.last_department_requested = department_requested.strip()
        
        if user_name and isinstance(user_name, str) and user_name.strip():
            session.context_data['user_name'] = user_name.strip()
        
        # Agregar datos adicionales al contexto
        for key, value in kwargs.items():
            session.context_data[key] = value
        
        logger.debug(f"Contexto actualizado para {user_id}: tipo={query_type}, mes={month_requested}, calendar_intent={calendar_intent}, time_ref={time_reference}, subject={subject_requested}, teacher={teacher_requested}, procedure={procedure_requested}, resource={resource_requested}")

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
        """Devuelve estadísticas detalladas del servicio de sesiones."""
        with self._lock:
            self._cleanup_expired_sessions()
            
            # Calcular estadísticas adicionales
            sessions_by_type = {}
            oldest_session_age = 0
            newest_session_age = float('inf')
            total_context_data_size = 0
            
            current_time = time.time()
            
            for user_id, session in self.sessions.items():
                # Estadísticas por tipo de consulta
                query_type = session.last_query_type or "unknown"
                sessions_by_type[query_type] = sessions_by_type.get(query_type, 0) + 1
                
                # Edad de las sesiones
                session_age = current_time - session.last_activity
                oldest_session_age = max(oldest_session_age, session_age)
                newest_session_age = min(newest_session_age, session_age)
                
                # Tamaño del contexto
                total_context_data_size += len(str(session.context_data))
            
            # Si no hay sesiones, ajustar valores
            if not self.sessions:
                newest_session_age = 0
            
            return {
                "active_sessions": len(self.sessions),
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds,
                "sessions_by_type": sessions_by_type,
                "oldest_session_age_seconds": round(oldest_session_age, 2),
                "newest_session_age_seconds": round(newest_session_age, 2),
                "average_context_size_bytes": total_context_data_size // len(self.sessions) if self.sessions else 0,
                "sweeper_active": self._sweeper_thread.is_alive() if self._sweeper_thread else False,
                "sweeper_interval_seconds": self._sweeper_interval
            }


class SessionServiceSingleton:
    """Singleton para el servicio de sesiones con soporte para testing."""
    _instance: Optional[SessionService] = None
    _lock = threading.Lock()
    _initialized = False
    
    @classmethod
    def get_instance(cls, **kwargs) -> SessionService:
        """
        Obtiene o crea la instancia singleton del SessionService.
        
        IMPORTANTE: Los parámetros solo se usan en la primera inicialización.
        Llamadas subsecuentes ignoran los parámetros y devuelven la instancia existente.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    # Solo permitir configuración en la primera inicialización
                    if cls._initialized and kwargs:
                        logger.warning(
                            "SessionServiceSingleton ya fue inicializado. "
                            "Los parámetros proporcionados serán ignorados. "
                            "Use reset_instance() para reinicializar con nuevos parámetros."
                        )
                    
                    if not cls._initialized:
                        # Primera inicialización - usar parámetros proporcionados o defaults
                        default_kwargs = {
                            'max_sessions': 1000,
                            'ttl_seconds': None,  # None para usar config
                            'enable_background_sweeper': True
                        }
                        default_kwargs.update(kwargs)
                        
                        cls._instance = SessionService(**default_kwargs)
                        cls._initialized = True
                        logger.info("SessionServiceSingleton inicializado por primera vez")
                    else:
                        # Si ya fue inicializado pero _instance es None (no debería pasar)
                        cls._instance = SessionService()
                        
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reinicia la instancia singleton (útil para testing)."""
        with cls._lock:
            if cls._instance:
                try:
                    cls._instance.stop()  # Detener sweeper si está activo
                except Exception as e:
                    logger.warning(f"Error al detener SessionService durante reset: {e}")
            cls._instance = None
            cls._initialized = False
            logger.info("SessionServiceSingleton reiniciado")
    
    @classmethod
    def set_instance(cls, instance: SessionService):
        """Establece una instancia específica (útil para testing con mocks)."""
        with cls._lock:
            if cls._instance:
                try:
                    cls._instance.stop()
                except Exception as e:
                    logger.warning(f"Error al detener SessionService anterior: {e}")
            cls._instance = instance
            cls._initialized = True
            logger.info("SessionServiceSingleton configurado con instancia personalizada")


# Función de conveniencia para obtener la instancia global
def get_session_service(**kwargs) -> SessionService:
    """Obtiene la instancia global del SessionService."""
    return SessionServiceSingleton.get_instance(**kwargs)


# Mantener compatibilidad hacia atrás
session_service = get_session_service()
