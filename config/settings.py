"""
Configuraciones del sistema RAG de la Facultad de Medicina UBA.
"""
import os
import logging
from pathlib import Path

# Crear directorio de logs si no existe
Path("logs").mkdir(exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Salida a consola
        logging.FileHandler(Path('logs') / 'app.log')  # Salida a archivo
    ]
)
logger = logging.getLogger(__name__)

# Nombre predeterminado de modelos para utilizar
PRIMARY_MODEL = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')
FALLBACK_MODEL = os.getenv('FALLBACK_MODEL', 'gpt-4.1-nano')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

# Límites para la API
MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '300'))
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))

# Parámetros de generación
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
TOP_P = float(os.getenv('TOP_P', '0.9'))
TOP_K = int(os.getenv('TOP_K', '50'))

# Configuración RAG
RAG_NUM_CHUNKS = int(os.getenv('RAG_NUM_CHUNKS', '5'))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
EMBEDDINGS_DIR = os.getenv('EMBEDDINGS_DIR', 'data/embeddings')

# Gestión del historial de conversación
MAX_HISTORY_LENGTH = 5

# Configuración para Google Cloud Storage
USE_GCS = os.getenv('GCS_BUCKET_NAME') is not None
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', '')

# Configuración para Google Sheets (Cursos)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
CURSOS_SPREADSHEET_ID = os.getenv('CURSOS_SPREADSHEET_ID', '')

# Configuración de dispositivo
DEVICE_PREF = os.getenv('DEVICE', 'auto') 