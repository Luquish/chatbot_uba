"""
Configuración específica para la lectura del calendario académico de la Facultad de Medicina UBA.
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de los calendarios
CALENDARS = {
    'examenes': {
        'id': os.getenv('CALENDAR_ID_EXAMENES'),
        'name': 'Exámenes',
        'description': 'Fechas de exámenes parciales, finales y recuperatorios',
        'keywords': ['examen', 'parcial', 'final', 'recuperatorio', 'coloquio', 'evaluación'],
        'color': '#4285f4'  # Color azul del calendario
    },
    'inscripciones': {
        'id': os.getenv('CALENDAR_ID_INSCRIPCIONES'),
        'name': 'Inscripciones',
        'description': 'Calendario oficial de inscripciones a materias, finales y cursadas',
        'keywords': ['inscripción', 'inscripciones', 'inscribir', 'anotarse', 'anotar', 'reasignación'],
        'color': '#f4b400'  # Color amarillo del calendario
    },
    'cursada': {
        'id': os.getenv('CALENDAR_ID_CURSADA'),
        'name': 'Cursada',
        'description': 'Inicio y fin de cursada, inicio y fin de vacaciones, etc.',
        'keywords': ['cuatrimestre', 'inicio', 'fin', 'final', 'comienzo', 'fin de cursada', 'vacaciones'],
        'color': '#a4725b'  # Color marrón del calendario
    },
    'tramites': {
        'id': os.getenv('CALENDAR_ID_TRAMITES'),
        'name': 'Trámites UBA Medicina',
        'description': 'Fechas importantes para trámites administrativos',
        'keywords': ['trámite', 'vencimiento', 'documentación', 'administrativo', 'reincorporación'],
        'color': '#7627bb'  # Color morado del calendario
    }
}

# Configuración general del calendario
CALENDAR_CONFIG = {
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