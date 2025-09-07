"""
Interfaces y implementaciones para persistencia de sesiones.
Permite extensibilidad futura con diferentes backends de almacenamiento.
"""

import json
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import asdict
from services.session_service import UserSession

logger = logging.getLogger(__name__)


class SessionPersistenceInterface(ABC):
    """Interface para diferentes backends de persistencia de sesiones."""
    
    @abstractmethod
    def save_session(self, user_id: str, session: UserSession) -> bool:
        """Guarda una sesión."""
        pass
    
    @abstractmethod
    def load_session(self, user_id: str) -> Optional[UserSession]:
        """Carga una sesión."""
        pass
    
    @abstractmethod
    def delete_session(self, user_id: str) -> bool:
        """Elimina una sesión."""
        pass
    
    @abstractmethod
    def cleanup_expired_sessions(self, ttl_seconds: int) -> int:
        """Limpia sesiones expiradas y retorna el número eliminado."""
        pass


class InMemorySessionPersistence(SessionPersistenceInterface):
    """Implementación en memoria (sin persistencia real)."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def save_session(self, user_id: str, session: UserSession) -> bool:
        try:
            self.sessions[user_id] = asdict(session)
            return True
        except Exception as e:
            logger.error(f"Error guardando sesión en memoria para {user_id}: {e}")
            return False
    
    def load_session(self, user_id: str) -> Optional[UserSession]:
        try:
            session_data = self.sessions.get(user_id)
            if session_data:
                return UserSession(**session_data)
            return None
        except Exception as e:
            logger.error(f"Error cargando sesión desde memoria para {user_id}: {e}")
            return None
    
    def delete_session(self, user_id: str) -> bool:
        try:
            if user_id in self.sessions:
                del self.sessions[user_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando sesión de memoria para {user_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self, ttl_seconds: int) -> int:
        try:
            current_time = time.time()
            expired_users = []
            
            for user_id, session_data in self.sessions.items():
                if current_time - session_data.get('last_activity', 0) > ttl_seconds:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.sessions[user_id]
            
            return len(expired_users)
        except Exception as e:
            logger.error(f"Error limpiando sesiones expiradas: {e}")
            return 0


class FileSessionPersistence(SessionPersistenceInterface):
    """Implementación con archivos JSON (útil para desarrollo/testing)."""
    
    def __init__(self, sessions_dir: str = "data/sessions"):
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)
    
    def _get_session_path(self, user_id: str) -> str:
        """Obtiene la ruta del archivo de sesión."""
        # Sanitizar user_id para uso como nombre de archivo
        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.sessions_dir, f"session_{safe_user_id}.json")
    
    def save_session(self, user_id: str, session: UserSession) -> bool:
        try:
            session_path = self._get_session_path(user_id)
            session_data = asdict(session)
            
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error guardando sesión en archivo para {user_id}: {e}")
            return False
    
    def load_session(self, user_id: str) -> Optional[UserSession]:
        try:
            session_path = self._get_session_path(user_id)
            
            if not os.path.exists(session_path):
                return None
            
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            return UserSession(**session_data)
        except Exception as e:
            logger.error(f"Error cargando sesión desde archivo para {user_id}: {e}")
            return None
    
    def delete_session(self, user_id: str) -> bool:
        try:
            session_path = self._get_session_path(user_id)
            if os.path.exists(session_path):
                os.remove(session_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando archivo de sesión para {user_id}: {e}")
            return False
    
    def cleanup_expired_sessions(self, ttl_seconds: int) -> int:
        try:
            current_time = time.time()
            expired_count = 0
            
            for filename in os.listdir(self.sessions_dir):
                if not filename.startswith("session_") or not filename.endswith(".json"):
                    continue
                
                session_path = os.path.join(self.sessions_dir, filename)
                try:
                    with open(session_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    if current_time - session_data.get('last_activity', 0) > ttl_seconds:
                        os.remove(session_path)
                        expired_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error procesando archivo de sesión {filename}: {e}")
            
            return expired_count
        except Exception as e:
            logger.error(f"Error limpiando archivos de sesiones expiradas: {e}")
            return 0


class PersistentSessionService:
    """
    SessionService extendido con capacidades de persistencia.
    
    Ejemplo de cómo extender el SessionService para soporte de persistencia.
    En el futuro se podría integrar directamente o usar como reemplazo.
    """
    
    def __init__(self, persistence: SessionPersistenceInterface, session_service):
        self.persistence = persistence
        self.session_service = session_service
        self._enabled = True
    
    def enable_persistence(self):
        """Habilita la persistencia."""
        self._enabled = True
    
    def disable_persistence(self):
        """Deshabilita la persistencia."""
        self._enabled = False
    
    def get_session(self, user_id: str):
        """Obtiene sesión con carga automática desde persistencia."""
        if self._enabled:
            # Intentar cargar desde persistencia primero
            persisted_session = self.persistence.load_session(user_id)
            if persisted_session and not persisted_session.is_expired(self.session_service.ttl_seconds):
                # Restaurar sesión en memoria
                self.session_service.sessions[user_id] = persisted_session
                return persisted_session
        
        # Usar comportamiento normal del SessionService
        return self.session_service.get_session(user_id)
    
    def update_session_context(self, user_id: str, **kwargs):
        """Actualiza contexto con guardado automático."""
        self.session_service.update_session_context(user_id, **kwargs)
        
        if self._enabled:
            session = self.session_service.sessions.get(user_id)
            if session:
                self.persistence.save_session(user_id, session)
    
    def cleanup_expired_sessions(self) -> int:
        """Limpia sesiones expiradas tanto en memoria como en persistencia."""
        # Limpiar en memoria
        memory_cleaned = len([
            user_id for user_id, session in self.session_service.sessions.items()
            if session.is_expired(self.session_service.ttl_seconds)
        ])
        self.session_service._cleanup_expired_sessions()
        
        # Limpiar en persistencia si está habilitada
        persistence_cleaned = 0
        if self._enabled:
            persistence_cleaned = self.persistence.cleanup_expired_sessions(self.session_service.ttl_seconds)
        
        total_cleaned = memory_cleaned + persistence_cleaned
        if total_cleaned > 0:
            logger.info(f"Sesiones expiradas eliminadas: {memory_cleaned} memoria + {persistence_cleaned} persistencia = {total_cleaned}")
        
        return total_cleaned


# Factory para crear diferentes tipos de persistencia
def create_session_persistence(persistence_type: str = "memory", **kwargs) -> SessionPersistenceInterface:
    """
    Factory para crear instancias de persistencia de sesiones.
    
    Args:
        persistence_type: Tipo de persistencia ("memory", "file", etc.)
        **kwargs: Argumentos específicos para cada tipo
    """
    if persistence_type == "memory":
        return InMemorySessionPersistence()
    elif persistence_type == "file":
        return FileSessionPersistence(**kwargs)
    else:
        raise ValueError(f"Tipo de persistencia no soportado: {persistence_type}")


# Función de utilidad para configurar persistencia en el futuro
def setup_session_persistence(session_service, persistence_type: str = "memory", **kwargs):
    """
    Configura persistencia para un SessionService existente.
    
    Returns:
        PersistentSessionService: Servicio extendido con persistencia
    """
    persistence = create_session_persistence(persistence_type, **kwargs)
    return PersistentSessionService(persistence, session_service)