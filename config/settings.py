"""
Configuración del sistema RAG de la Facultad de Medicina UBA.
Optimizado para el backend del chatbot.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# =============================================================================
# CONFIGURACIÓN DE LOGGING
# =============================================================================

# Crear directorio de logs si no existe
Path("logs").mkdir(exist_ok=True)

# Configuración de logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(levelname)s - %(message)s')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),  # Salida a consola
        logging.FileHandler(Path('logs') / 'app.log')  # Salida a archivo
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN DE OPENAI
# =============================================================================

# API Keys y Modelos
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PRIMARY_MODEL = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
FALLBACK_MODEL = os.getenv('FALLBACK_MODEL', 'gpt-4.1-nano')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

# Parámetros de generación
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
TOP_P = float(os.getenv('TOP_P', '0.9'))
TOP_K = int(os.getenv('TOP_K', '50'))
MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '300'))
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))

# =============================================================================
# CONFIGURACIÓN DE RAG
# =============================================================================

# Parámetros de RAG
RAG_NUM_CHUNKS = int(os.getenv('RAG_NUM_CHUNKS', '8'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.2'))
EMBEDDINGS_DIR = os.getenv('EMBEDDINGS_DIR', 'data/embeddings')

# Gestión del historial de conversación
MAX_HISTORY_LENGTH = int(os.getenv('MAX_HISTORY_LENGTH', '5'))

# =============================================================================
# CONFIGURACIÓN DE GOOGLE CLOUD STORAGE
# =============================================================================

# Google Cloud Storage
USE_GCS = os.getenv('USE_GCS', 'true').lower() == 'true' and os.getenv('GCS_BUCKET_NAME') is not None
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', '')
GCS_AUTO_REFRESH = os.getenv('GCS_AUTO_REFRESH', 'true').lower() == 'true'
GCS_REFRESH_INTERVAL = int(os.getenv('GCS_REFRESH_INTERVAL', '300'))  # 5 minutos

# =============================================================================
# CONFIGURACIÓN DE WHATSAPP BUSINESS API
# =============================================================================

# WhatsApp Business API
WHATSAPP_API_TOKEN = os.getenv('WHATSAPP_API_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID')
WHATSAPP_WEBHOOK_VERIFY_TOKEN = os.getenv('WHATSAPP_WEBHOOK_VERIFY_TOKEN')

# Número de teléfono para pruebas
MY_PHONE_NUMBER = os.getenv('MY_PHONE_NUMBER')

# =============================================================================
# CONFIGURACIÓN DE GOOGLE APIS
# =============================================================================

# Google API Key (para Calendar y Sheets)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

# Google Sheets (Cursos)
CURSOS_SPREADSHEET_ID = os.getenv('CURSOS_SPREADSHEET_ID', '')

# Google Calendar (IDs de calendarios)
CALENDAR_ID_EXAMENES = os.getenv('CALENDAR_ID_EXAMENES')
CALENDAR_ID_INSCRIPCIONES = os.getenv('CALENDAR_ID_INSCRIPCIONES')
CALENDAR_ID_CURSADA = os.getenv('CALENDAR_ID_CURSADA')
CALENDAR_ID_TRAMITES = os.getenv('CALENDAR_ID_TRAMITES')

# =============================================================================
# CONFIGURACIÓN DEL SERVIDOR
# =============================================================================

# Configuración del servidor
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8080'))

# =============================================================================
# CONFIGURACIÓN DE DISPOSITIVO
# =============================================================================

# Configuración de dispositivo (principalmente para desarrollo)
DEVICE_PREF = os.getenv('DEVICE', 'auto')

# =============================================================================
# CONFIGURACIÓN DE INTEGRACIÓN
# =============================================================================

# Configuración para integración con drcecim_upload
DRCECIM_UPLOAD_INTEGRATION = os.getenv('DRCECIM_UPLOAD_INTEGRATION', 'true').lower() == 'true'

# =============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# =============================================================================

def validate_config():
    """
    Valida que todas las variables de entorno críticas estén configuradas.
    """
    missing_vars = []
    
    # Variables críticas
    if not OPENAI_API_KEY:
        missing_vars.append('OPENAI_API_KEY')
    
    # Variables de WhatsApp (opcionales pero necesarias para funcionalidad completa)
    whatsapp_vars = [
        ('WHATSAPP_API_TOKEN', WHATSAPP_API_TOKEN),
        ('WHATSAPP_PHONE_NUMBER_ID', WHATSAPP_PHONE_NUMBER_ID),
        ('WHATSAPP_BUSINESS_ACCOUNT_ID', WHATSAPP_BUSINESS_ACCOUNT_ID),
        ('WHATSAPP_WEBHOOK_VERIFY_TOKEN', WHATSAPP_WEBHOOK_VERIFY_TOKEN)
    ]
    
    missing_whatsapp = [var_name for var_name, var_value in whatsapp_vars if not var_value]
    if missing_whatsapp:
        logger.warning(f"Variables de WhatsApp no configuradas: {', '.join(missing_whatsapp)}")
        logger.warning("El chatbot funcionará pero no podrá enviar/recibir mensajes de WhatsApp")
    
    if missing_vars:
        raise ValueError(f"Las siguientes variables de entorno son requeridas: {', '.join(missing_vars)}")
    
    return True

# =============================================================================
# EXPORT DE CONFIGURACIÓN
# =============================================================================

# Diccionario con toda la configuración para fácil acceso
CONFIG = {
    'openai': {
        'api_key': OPENAI_API_KEY,
        'primary_model': PRIMARY_MODEL,
        'fallback_model': FALLBACK_MODEL,
        'embedding_model': EMBEDDING_MODEL,
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'top_k': TOP_K,
        'max_output_tokens': MAX_OUTPUT_TOKENS,
        'api_timeout': API_TIMEOUT,
    },
    'rag': {
        'num_chunks': RAG_NUM_CHUNKS,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'embeddings_dir': EMBEDDINGS_DIR,
        'max_history_length': MAX_HISTORY_LENGTH,
    },
    'gcs': {
        'use_gcs': USE_GCS,
        'bucket_name': GCS_BUCKET_NAME,
        'auto_refresh': GCS_AUTO_REFRESH,
        'refresh_interval': GCS_REFRESH_INTERVAL,
    },
    'whatsapp': {
        'api_token': WHATSAPP_API_TOKEN,
        'phone_number_id': WHATSAPP_PHONE_NUMBER_ID,
        'business_account_id': WHATSAPP_BUSINESS_ACCOUNT_ID,
        'webhook_verify_token': WHATSAPP_WEBHOOK_VERIFY_TOKEN,
        'my_phone_number': MY_PHONE_NUMBER,
    },
    'google_apis': {
        'api_key': GOOGLE_API_KEY,
        'cursos_spreadsheet_id': CURSOS_SPREADSHEET_ID,
        'calendar_examenes': CALENDAR_ID_EXAMENES,
        'calendar_inscripciones': CALENDAR_ID_INSCRIPCIONES,
        'calendar_cursada': CALENDAR_ID_CURSADA,
        'calendar_tramites': CALENDAR_ID_TRAMITES,
    },
    'server': {
        'environment': ENVIRONMENT,
        'host': HOST,
        'port': PORT,
    },
    'system': {
        'device_pref': DEVICE_PREF,
        'drcecim_upload_integration': DRCECIM_UPLOAD_INTEGRATION,
        'log_level': LOG_LEVEL,
        'log_format': LOG_FORMAT,
    }
}

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

# Ejecutar validación al importar (solo si no es el módulo principal)
if __name__ != '__main__':
    validate_config()
    logger.info(f"Configuración cargada para entorno: {ENVIRONMENT}")
    logger.info(f"Usando GCS: {USE_GCS}")
    logger.info(f"WhatsApp configurado: {bool(WHATSAPP_API_TOKEN)}")
    logger.info(f"Google APIs configuradas: {bool(GOOGLE_API_KEY)}") 