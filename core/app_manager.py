"""
Administrador de aplicación para manejar el estado global de manera centralizada.
Elimina variables globales y proporciona un patrón singleton.
"""
import os
import logging
from typing import Optional
from rag_system import RAGSystem
from handlers.telegram_handler import TelegramHandler

logger = logging.getLogger(__name__)


class AppManager:
    """Administrador centralizado de la aplicación."""
    
    _instance: Optional['AppManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.rag_system: Optional[RAGSystem] = None
        self._rag_initialized = False
        self._telegram_handler: Optional[TelegramHandler] = None
        
        # Configuración de ambiente
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_webhook_secret = os.getenv('TELEGRAM_WEBHOOK_SECRET')
        self.telegram_admin_user_id = os.getenv('TELEGRAM_ADMIN_USER_ID')
        
        logger.info(f"AppManager inicializado para entorno: {self.environment}")
        self._initialized = True
    
    def get_rag_system(self) -> RAGSystem:
        """
        Obtiene una instancia del sistema RAG, inicializándola si es necesario.
        
        Returns:
            RAGSystem: Instancia del sistema RAG
        """
        if not self._rag_initialized:
            self._initialize_rag_system()
        
        if self.rag_system is None:
            raise RuntimeError("No se pudo inicializar el sistema RAG")
        
        return self.rag_system
    
    def _initialize_rag_system(self):
        """Inicializa el sistema RAG de manera lazy."""
        try:
            logger.info("Inicializando sistema RAG...")
            self.rag_system = RAGSystem()
            self._rag_initialized = True
            logger.info("✅ Sistema RAG inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema RAG: {str(e)}")
            self.rag_system = None
            self._rag_initialized = False
            raise
    
    def get_telegram_handler(self) -> TelegramHandler:
        """
        Obtiene una instancia del handler de Telegram.
        
        Returns:
            TelegramHandler: Handler de Telegram
            
        Raises:
            RuntimeError: Si no hay token de bot configurado
        """
        if not self.telegram_bot_token:
            raise RuntimeError("Integración con Telegram no disponible. Falta TELEGRAM_BOT_TOKEN.")
        
        if self._telegram_handler is None:
            self._telegram_handler = TelegramHandler(self.telegram_bot_token)
            logger.info("✅ Handler de Telegram inicializado")
        
        return self._telegram_handler
    
    def is_rag_ready(self) -> bool:
        """
        Verifica si el sistema RAG está listo para usar.
        
        Returns:
            bool: True si el RAG está inicializado y listo
        """
        return self._rag_initialized and self.rag_system is not None
    
    def is_telegram_ready(self) -> bool:
        """
        Verifica si Telegram está configurado.
        
        Returns:
            bool: True si hay token de bot disponible
        """
        return bool(self.telegram_bot_token)
    
    def restart_rag_system(self):
        """
        Reinicia el sistema RAG.
        Útil para actualizaciones en caliente o recuperación de errores.
        """
        logger.info("Reiniciando sistema RAG...")
        self.rag_system = None
        self._rag_initialized = False
        self._initialize_rag_system()
    
    def get_system_status(self) -> dict:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            dict: Estado de los componentes del sistema
        """
        return {
            "environment": self.environment,
            "rag_initialized": self._rag_initialized,
            "rag_ready": self.is_rag_ready(),
            "telegram_configured": self.is_telegram_ready(),
            "telegram_handler_ready": self._telegram_handler is not None
        }


# Instancia global (singleton)
app_manager = AppManager()