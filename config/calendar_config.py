"""
Configuración específica para la lectura del calendario académico de la Facultad de Medicina UBA.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración del calendario
CALENDAR_CONFIG = {
    'GOOGLE_CALENDAR_ID': os.getenv('GOOGLE_CALENDAR_ID'),
    'GOOGLE_CALENDAR_API_KEY': os.getenv('GOOGLE_CALENDAR_API_KEY'),
    'BASE_URL': 'https://www.googleapis.com/calendar/v3',
    'MAX_RESULTS': 10,
    'TIMEZONE': 'America/Argentina/Buenos_Aires'
}

# Formato de fechas para respuestas
DATE_FORMAT = {
    'DATETIME': '%A %d/%m/%Y %H:%M',
    'DATE': '%A %d/%m/%Y'
}