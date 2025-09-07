"""
Configuración del sistema RAG de la Facultad de Medicina UBA usando Pydantic.
Optimizado para el backend del chatbot.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class LoggingSettings(BaseSettings):
    """Configuración de logging."""
    
    log_level: str = Field(default='INFO', env='LOG_LEVEL')
    log_format: str = Field(default='%(levelname)s - %(message)s', env='LOG_FORMAT')
    
    class Config:
        env_prefix = ''

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        return v.upper() if v.upper() in valid_levels else 'INFO'


class OpenAISettings(BaseSettings):
    """Configuración de OpenAI API."""
    
    openai_api_key: Optional[str] = Field(default=None, env='OPENAI_API_KEY')
    primary_model: str = Field(default='gpt-4o-mini', env='PRIMARY_MODEL')
    fallback_model: str = Field(default='gpt-4.1-nano', env='FALLBACK_MODEL')
    embedding_model: str = Field(default='text-embedding-3-small', env='EMBEDDING_MODEL')
    
    # Parámetros de generación
    temperature: float = Field(default=0.7, env='TEMPERATURE')
    top_p: float = Field(default=0.9, env='TOP_P')
    top_k: int = Field(default=50, env='TOP_K')
    max_output_tokens: int = Field(default=300, env='MAX_OUTPUT_TOKENS')
    api_timeout: int = Field(default=30, env='API_TIMEOUT')
    
    class Config:
        env_prefix = ''


class RAGSettings(BaseSettings):
    """Configuración del sistema RAG."""
    
    rag_num_chunks: int = Field(default=8, env='RAG_NUM_CHUNKS')
    similarity_threshold: float = Field(default=-0.1, env='SIMILARITY_THRESHOLD')
    embeddings_dir: str = Field(default='data/embeddings', env='EMBEDDINGS_DIR')
    max_history_length: int = Field(default=5, env='MAX_HISTORY_LENGTH')
    
    class Config:
        env_prefix = ''


class GoogleCloudSettings(BaseSettings):
    """Configuración de Google Cloud Storage."""
    
    use_gcs: bool = Field(default=True, env='USE_GCS')
    gcs_bucket_name: str = Field(default='', env='GCS_BUCKET_NAME')
    gcs_auto_refresh: bool = Field(default=True, env='GCS_AUTO_REFRESH')
    gcs_refresh_interval: int = Field(default=300, env='GCS_REFRESH_INTERVAL')  # 5 minutos
    
    class Config:
        env_prefix = ''

    @field_validator('use_gcs', mode='before')
    @classmethod
    def validate_use_gcs(cls, v):
        if isinstance(v, str):
            return v.lower() == 'true' and bool(os.getenv('GCS_BUCKET_NAME'))
        return v


class TelegramSettings(BaseSettings):
    """Configuración de Telegram Bot API."""
    
    telegram_bot_token: Optional[str] = Field(default=None, env='TELEGRAM_BOT_TOKEN')
    telegram_webhook_secret: Optional[str] = Field(default=None, env='TELEGRAM_WEBHOOK_SECRET')
    telegram_admin_user_id: Optional[str] = Field(default=None, env='TELEGRAM_ADMIN_USER_ID')
    
    class Config:
        env_prefix = ''


class GoogleApisSettings(BaseSettings):
    """Configuración de Google APIs (Calendar y Sheets)."""
    
    google_api_key: str = Field(default='', env='GOOGLE_API_KEY')
    cursos_spreadsheet_id: str = Field(default='', env='CURSOS_SPREADSHEET_ID')
    
    # Google Calendar IDs
    calendar_id_examenes: Optional[str] = Field(default=None, env='CALENDAR_ID_EXAMENES')
    calendar_id_inscripciones: Optional[str] = Field(default=None, env='CALENDAR_ID_INSCRIPCIONES')
    calendar_id_cursada: Optional[str] = Field(default=None, env='CALENDAR_ID_CURSADA')
    calendar_id_tramites: Optional[str] = Field(default=None, env='CALENDAR_ID_TRAMITES')
    

    
    class Config:
        env_prefix = ''


class ServerSettings(BaseSettings):
    """Configuración del servidor."""
    
    environment: str = Field(default='development', env='ENVIRONMENT')
    host: str = Field(default='0.0.0.0', env='HOST')
    port: int = Field(default=8080, env='PORT')
    
    class Config:
        env_prefix = ''


class CloudSQLSettings(BaseSettings):
    """Configuración de Cloud SQL para PostgreSQL con pgvector."""
    
    db_user: str = Field(default='raguser', env='DB_USER')
    db_pass: str = Field(default='', env='DB_PASS')
    db_name: str = Field(default='ragdb', env='DB_NAME')
    cloud_sql_connection_name: Optional[str] = Field(default=None, env='CLOUD_SQL_CONNECTION_NAME')
    db_private_ip: bool = Field(default=False, env='DB_PRIVATE_IP')
    
    # Configuración para PostgreSQL local (desarrollo)
    db_host: str = Field(default='localhost', env='DB_HOST')
    db_port: int = Field(default=5432, env='DB_PORT')
    
    class Config:
        env_prefix = ''

    @field_validator('db_private_ip', mode='before')
    @classmethod
    def validate_db_private_ip(cls, v):
        if isinstance(v, str):
            return v.lower() == 'true'
        return v


class SystemSettings(BaseSettings):
    """Configuración del sistema."""
    
    device_pref: str = Field(default='auto', env='DEVICE')
    drcecim_upload_integration: bool = Field(default=True, env='DRCECIM_UPLOAD_INTEGRATION')
    
    class Config:
        env_prefix = ''

    @field_validator('drcecim_upload_integration', mode='before')
    @classmethod
    def validate_drcecim_integration(cls, v):
        if isinstance(v, str):
            return v.lower() == 'true'
        return v


class SessionSettings(BaseSettings):
    """Configuración de sesiones conversacionales efímeras."""
    
    ttl_seconds: int = Field(default=1800, env='SESSION_TTL_SECONDS')  # 30 minutos
    sweeper_interval_seconds: int = Field(default=60, env='SESSION_SWEEPER_INTERVAL')  # 1 minuto
    
    class Config:
        env_prefix = ''


class ChatbotConfig(BaseSettings):
    """Configuración principal del chatbot que agrupa todas las configuraciones."""
    
    # Subsecciones de configuración
    logging: LoggingSettings = LoggingSettings()
    openai: OpenAISettings = OpenAISettings()
    rag: RAGSettings = RAGSettings()
    gcs: GoogleCloudSettings = GoogleCloudSettings()
    telegram: TelegramSettings = TelegramSettings()
    google_apis: GoogleApisSettings = GoogleApisSettings()
    server: ServerSettings = ServerSettings()
    system: SystemSettings = SystemSettings()
    session: SessionSettings = SessionSettings()
    cloudsql: CloudSQLSettings = CloudSQLSettings()
    
    class Config:
        env_prefix = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_logging()
        self._create_directories()

    def _setup_logging(self):
        """Configura el sistema de logging."""
        # Crear directorio de logs si no existe
        Path("logs").mkdir(exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, self.logging.log_level.upper()),
            format=self.logging.log_format,
            handlers=[
                logging.StreamHandler(),  # Salida a consola
                logging.FileHandler(Path('logs') / 'app.log')  # Salida a archivo
            ]
        )

    def _create_directories(self):
        """Crear los directorios necesarios para el funcionamiento del sistema."""
        dirs_to_create = [
            self.rag.embeddings_dir,
            'logs'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def validate_config(self):
        """Valida que todas las variables de entorno críticas estén configuradas."""
        missing_vars = []
        
        # Variables críticas
        if not self.openai.openai_api_key:
            missing_vars.append('OPENAI_API_KEY')
        
        # Variables de Cloud SQL (críticas para RAG)
        if not self.cloudsql.db_user or not self.cloudsql.db_pass or not self.cloudsql.db_name:
            missing_vars.extend(['DB_USER', 'DB_PASS', 'DB_NAME'])
        
        # Variables de Telegram (opcionales pero necesarias para funcionalidad completa)
        telegram_vars = [
            ('TELEGRAM_BOT_TOKEN', self.telegram.telegram_bot_token),
            ('TELEGRAM_WEBHOOK_SECRET', self.telegram.telegram_webhook_secret),
            ('TELEGRAM_ADMIN_USER_ID', self.telegram.telegram_admin_user_id)
        ]
        
        missing_telegram = [var_name for var_name, var_value in telegram_vars if not var_value]
        if missing_telegram:
            logger.warning(f"Variables de Telegram no configuradas: {', '.join(missing_telegram)}")
            logger.warning("El chatbot funcionará pero no podrá enviar/recibir mensajes de Telegram")
        
        if missing_vars:
            raise ValueError(f"Las siguientes variables de entorno son requeridas: {', '.join(missing_vars)}")
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a un diccionario para compatibilidad."""
        return {
            'openai': {
                'api_key': self.openai.openai_api_key,
                'primary_model': self.openai.primary_model,
                'fallback_model': self.openai.fallback_model,
                'embedding_model': self.openai.embedding_model,
                'temperature': self.openai.temperature,
                'top_p': self.openai.top_p,
                'top_k': self.openai.top_k,
                'max_output_tokens': self.openai.max_output_tokens,
                'api_timeout': self.openai.api_timeout,
            },
            'rag': {
                'num_chunks': self.rag.rag_num_chunks,
                'similarity_threshold': self.rag.similarity_threshold,
                'embeddings_dir': self.rag.embeddings_dir,
                'max_history_length': self.rag.max_history_length,
            },
            'gcs': {
                'use_gcs': self.gcs.use_gcs,
                'bucket_name': self.gcs.gcs_bucket_name,
                'auto_refresh': self.gcs.gcs_auto_refresh,
                'refresh_interval': self.gcs.gcs_refresh_interval,
            },
            'telegram': {
                'bot_token': self.telegram.telegram_bot_token,
                'webhook_secret': self.telegram.telegram_webhook_secret,
                'admin_user_id': self.telegram.telegram_admin_user_id,
            },
            'google_apis': {
                'api_key': self.google_apis.google_api_key,
                'cursos_spreadsheet_id': self.google_apis.cursos_spreadsheet_id,
                'calendar_examenes': self.google_apis.calendar_id_examenes,
                'calendar_inscripciones': self.google_apis.calendar_id_inscripciones,
                'calendar_cursada': self.google_apis.calendar_id_cursada,
                'calendar_tramites': self.google_apis.calendar_id_tramites,

            },
            'server': {
                'environment': self.server.environment,
                'host': self.server.host,
                'port': self.server.port,
            },
            'system': {
                'device_pref': self.system.device_pref,
                'drcecim_upload_integration': self.system.drcecim_upload_integration,
                'log_level': self.logging.log_level,
                'log_format': self.logging.log_format,
            },
            'cloudsql': {
                'db_user': self.cloudsql.db_user,
                'db_pass': self.cloudsql.db_pass,
                'db_name': self.cloudsql.db_name,
                'cloud_sql_connection_name': self.cloudsql.cloud_sql_connection_name,
                'db_private_ip': self.cloudsql.db_private_ip,
                'db_host': self.cloudsql.db_host,
                'db_port': self.cloudsql.db_port,
            }
        }


# =============================================================================
# INSTANCIA GLOBAL DE CONFIGURACIÓN
# =============================================================================

# Instancia global de configuración
config = ChatbotConfig()

# Logger configurado
logger = logging.getLogger(__name__)

# =============================================================================
# VARIABLES DE COMPATIBILIDAD LEGACY
# =============================================================================

# Logging
LOG_LEVEL = config.logging.log_level
LOG_FORMAT = config.logging.log_format

# OpenAI
OPENAI_API_KEY = config.openai.openai_api_key
PRIMARY_MODEL = config.openai.primary_model
FALLBACK_MODEL = config.openai.fallback_model
EMBEDDING_MODEL = config.openai.embedding_model
TEMPERATURE = config.openai.temperature
TOP_P = config.openai.top_p
TOP_K = config.openai.top_k
MAX_OUTPUT_TOKENS = config.openai.max_output_tokens
API_TIMEOUT = config.openai.api_timeout

# RAG
RAG_NUM_CHUNKS = config.rag.rag_num_chunks
SIMILARITY_THRESHOLD = config.rag.similarity_threshold
EMBEDDINGS_DIR = config.rag.embeddings_dir
MAX_HISTORY_LENGTH = config.rag.max_history_length

# Google Cloud Storage
USE_GCS = config.gcs.use_gcs
GCS_BUCKET_NAME = config.gcs.gcs_bucket_name
GCS_AUTO_REFRESH = config.gcs.gcs_auto_refresh
GCS_REFRESH_INTERVAL = config.gcs.gcs_refresh_interval

# Telegram
TELEGRAM_BOT_TOKEN = config.telegram.telegram_bot_token
TELEGRAM_WEBHOOK_SECRET = config.telegram.telegram_webhook_secret
TELEGRAM_ADMIN_USER_ID = config.telegram.telegram_admin_user_id

# Google APIs
GOOGLE_API_KEY = config.google_apis.google_api_key
CURSOS_SPREADSHEET_ID = config.google_apis.cursos_spreadsheet_id
CALENDAR_ID_EXAMENES = config.google_apis.calendar_id_examenes
CALENDAR_ID_INSCRIPCIONES = config.google_apis.calendar_id_inscripciones
CALENDAR_ID_CURSADA = config.google_apis.calendar_id_cursada
CALENDAR_ID_TRAMITES = config.google_apis.calendar_id_tramites

# Server
ENVIRONMENT = config.server.environment
HOST = config.server.host
PORT = config.server.port

# System
DEVICE_PREF = config.system.device_pref
DRCECIM_UPLOAD_INTEGRATION = config.system.drcecim_upload_integration

# Cloud SQL
DB_USER = config.cloudsql.db_user
DB_PASS = config.cloudsql.db_pass
DB_NAME = config.cloudsql.db_name
CLOUD_SQL_CONNECTION_NAME = config.cloudsql.cloud_sql_connection_name
DB_PRIVATE_IP = config.cloudsql.db_private_ip
DB_HOST = config.cloudsql.db_host
DB_PORT = config.cloudsql.db_port

# =============================================================================
# EXPORT DE CONFIGURACIÓN PARA COMPATIBILIDAD
# =============================================================================

# Diccionario con toda la configuración para fácil acceso (legacy)
CONFIG = config.to_dict()

# Función de validación legacy
def validate_config():
    """Función legacy de validación de configuración."""
    return config.validate_config()

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

# Ejecutar validación al importar (solo si no es el módulo principal)
if __name__ != '__main__':
    validate_config()
    logger.info(f"Configuración cargada para entorno: {ENVIRONMENT}")
    logger.info(f"Usando GCS: {USE_GCS}")
    logger.info(f"Telegram configurado: {bool(TELEGRAM_BOT_TOKEN)}")
    logger.info(f"Google APIs configuradas: {bool(GOOGLE_API_KEY)}") 