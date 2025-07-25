"""
Constantes utilizadas por el sistema RAG de la Facultad de Medicina UBA.
"""
import os
import random
from typing import Dict, List

# Nueva estructura de intenciones para clasificación semántica
INTENT_EXAMPLES = {
    'saludo': {
        'examples': [
            "hola",
            "buenos días",
            "qué tal",
            "buenas"
        ],
        'context': "El usuario está iniciando la conversación o saludando"
    },
    'pregunta_nombre': {
        'examples': [
            "como me llamo yo",
            "cual es mi nombre",
            "sabes como me llamo",
            "como sabes mi nombre",
            "por que sabes mi nombre",
            "de donde sacaste mi nombre",
            "como conseguiste mi nombre",
            "por que conoces mi nombre"
        ],
        'context': "El usuario pregunta específicamente sobre cómo conocemos su nombre"
    },
    'cortesia': {
        'examples': [
            "cómo estás",
            "como estas",
            "como te sentis",
            "como va",
            "todo bien"
        ],
        'context': "El usuario hace una pregunta de cortesía"
    },
    'referencia_anterior': {
        'examples': [
            "resúmeme eso",
            "resumeme eso",
            "puedes resumir lo anterior",
            "podrias resumirme el mensaje anterior",
            "podrias resumirme ese texto",
            "resume el mensaje anterior",
            "explica de nuevo",
            "explícame eso",
            "explicame eso de nuevo",
            "acorta esa explicación",
            "simplifica lo que dijiste",
            "decimelo más corto",
            "decime mas corto",
            "lo pode abreviar?",
            "puedes hacer un resumen"
        ],
        'context': "El usuario está pidiendo un resumen o clarificación del mensaje anterior"
    },
    'pregunta_capacidades': {
        'examples': [
            "qué podés hacer",
            "en qué me podés ayudar",
            "para qué servís",
            "qué tipo de consultas puedo hacer",
            "que sabes",
            "que sabes hacer",
            "que podes hacer",
            "que me podes decir",
            "cuales son tus funciones",
            "que funciones tenes",
            "que te puedo preguntar",
            "que puedo preguntarte",
            "que tipos de dudas puedo consultar",
            "en que me podes ayudar"
        ],
        'context': "El usuario quiere saber las capacidades del bot"
    },
    'identidad': {
        'examples': [
            "quién sos",
            "cómo te llamás",
            "sos un bot",
            "sos una persona"
        ],
        'context': "El usuario pregunta sobre la identidad del bot"
    },
    'consulta_administrativa': {
        'examples': [
            "cómo hago un trámite",
            "necesito una constancia",
            "dónde presento documentación",
            "quiero dar de baja una materia",
            "cuántas materias debo aprobar",
            "en cuánto tiempo tengo que terminar la carrera",
            "cómo se define el año académico",
            "qué derechos tengo para inscribirme",
            # Nuevos ejemplos sobre denuncias y trámites administrativos
            "cómo presento una denuncia",
            "qué tengo que hacer para presentar una denuncia",
            "dónde puedo hacer una queja formal",
            "procedimiento para reportar un problema",
            "cómo puedo denunciar una situación irregular",
            "pasos para hacer una denuncia",
            "dónde se presentan las quejas",
            "quiero denunciar a alguien, qué hago",
            "cómo inicio un reclamo formal",
            "quiero reportar una irregularidad",
            "cómo suspender temporalmente mi condición de alumno",
            "puedo pedir suspensión de mis estudios",
            "qué pasa si pierdo la regularidad",
            "cómo solicito readmisión"
        ],
        'context': "El usuario necesita información sobre trámites administrativos, condiciones de regularidad o procedimientos formales"
    },
    'consulta_academica': {
        'examples': [
            "cuándo es el parcial",
            "dónde encuentro el programa",
            "cómo es la cursada",
            "qué necesito para aprobar",
            "cómo se evalúa la calidad de enseñanza",
            "qué materias puedo cursar",
            # Nuevos ejemplos relacionados con cuestiones académicas
            "cuántas materias tengo que aprobar para mantener regularidad",
            "qué pasa si tengo muchos aplazos",
            "cuántos aplazos puedo tener como máximo",
            "cuál es el porcentaje máximo de aplazos permitido",
            "en cuánto tiempo tengo que terminar la carrera",
            "plazo máximo para completar mis estudios",
            "cómo saber si soy alumno regular",
            "qué derechos tengo como alumno",
            "qué pasa si no apruebo suficientes materias",
            "cómo puedo cursar materias en otra facultad"
        ],
        'context': "El usuario necesita información académica sobre cursada, evaluación y aprobación"
    },
    'consulta_medica': {
        'examples': [
            "me duele la cabeza",
            "tengo síntomas de",
            "dónde puedo consultar por un dolor",
            "necesito un diagnóstico",
            "tengo fiebre"
        ],
        'context': "El usuario hace una consulta médica que no podemos responder"
    },
    'consulta_reglamento': {
        'examples': [
            "qué dice el reglamento sobre",
            "está permitido",
            "cuáles son las normas",
            "qué pasa si no cumplo",
            "qué medidas toman si me porto mal",
            "quién decide las sanciones",
            "qué castigos hay",
            "para qué sirven las medidas disciplinarias",
            "qué sanciones aplican",
            "si cometo una falta",
            "quién evalúa mi comportamiento",
            "qué pasa si rompo las reglas",
            # Nuevos ejemplos relacionados con el régimen disciplinario
            "qué sanciones hay si agredo a un profesor",
            "qué pasa si me comporto mal en la facultad",
            "cuáles son las sanciones disciplinarias",
            "quién puede denunciar una falta disciplinaria",
            "cómo es el proceso de un sumario disciplinario",
            "qué pasa si me suspenden preventivamente",
            "puedo apelar una sanción",
            "por cuánto tiempo pueden suspenderme",
            "qué pasa si falsifiqué un documento",
            "qué sucede si agravo a otro estudiante",
            "cuánto dura la suspensión por falta de respeto",
            "qué es un apercibimiento",
            "puedo estudiar en otra facultad si me suspenden",
            "cómo se presenta una denuncia por conducta inapropiada",
            "qué ocurre si adulteré un acta de examen"
        ],
        'context': "El usuario pregunta sobre normativas, reglamentos y medidas disciplinarias"
    },
    'agradecimiento': {
        'examples': [
            "perfecto",
            "gracias",
            "ok",
            "okk",
            "okey",
            "okay",
            "dale",
            "listo",
            "entendido",
            "genial",
            "excelente",
            "bárbaro",
            "buenísimo",
            "joya"
        ],
        'context': "El usuario agradece o confirma que entendió la información"
    }
}

GREETING_WORDS = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches', 'buen dia', 'saludos', 'que tal']

# Lista de emojis para enriquecer las respuestas
information_emojis = ["📚", "📖", "ℹ️", "📊", "🔍", "📝", "📋", "📈", "📌", "🧠"]
greeting_emojis = ["👋", "😊", "🤓", "👨‍⚕️", "👩‍⚕️", "🎓", "🌟"]
warning_emojis = ["⚠️", "❗", "⚡", "🚨"]
success_emojis = ["✅", "💫", "🎉", "💡"]
medical_emojis = ["🏥", "👨‍⚕️", "👩‍⚕️", "🩺"]

# Configuraciones específicas para consultas de calendario
CALENDAR_INTENT_MAPPING = {
    'examenes': {
        'keywords': ['examen', 'examenes', 'parcial', 'parciales', 'final', 'finales', 'evaluación'],
        'tool': 'get_events_by_type',
        'params': {'calendar_type': 'examenes'},
        'no_events_message': 'No hay exámenes programados en este momento.'
    },
    'inscripciones': {
        'keywords': ['inscripción', 'inscripciones', 'inscribir', 'anotar', 'anotarse', 'reasignación'],
        'tool': 'get_events_by_type',
        'params': {'calendar_type': 'inscripciones'},
        'no_events_message': 'No hay eventos de inscripción programados en este momento.'
    },
    'cursada': {
        'keywords': ['cursada', 'cursadas', 'cuatrimestre', 'inicio', 'fin', 'final', 'comienzo', 'fin de cursada', 'vacaciones'],
        'tool': 'get_events_by_type',
        'params': {'calendar_type': 'cursada'},
        'no_events_message': 'No hay información sobre cursadas en este momento.'
    },
    'tramites': {
        'keywords': ['trámite', 'tramite', 'trámites', 'tramites', 'documentación', 'documentacion', 'administrativo'],
        'tool': 'get_events_by_type',
        'params': {'calendar_type': 'tramites'},
        'no_events_message': 'No hay trámites programados en este momento.'
    }
}

CALENDAR_MESSAGES = {
    'NO_EVENTS': 'No encontré eventos programados para ese período en el calendario académico.',
    'ERROR_FETCH': 'Lo siento, no pude acceder al calendario académico en este momento. Por favor, intentá más tarde. Mientras tanto te podes comunicar con @cecim.nemed por instagram',
    'MULTIPLE_EVENTS': 'Encontré varios eventos que coinciden con tu búsqueda:',
    'NO_SPECIFIC_EVENT': 'No encontré eventos específicos que coincidan con tu búsqueda en el calendario.',
    'PAST_EVENT': 'Ese evento ya pasó. ¿Querés que te muestre los próximos eventos similares?'
}

CALENDAR_SEARCH_CONFIG = {
    'MAX_RESULTS': 5,  # Número máximo de resultados a devolver
    'DEFAULT_TIMESPAN': 30,  # Días hacia adelante por defecto
    'MAX_TIMESPAN': 180,  # Máximo número de días hacia adelante para buscar
    'TIME_MIN': 'now',  # Comenzar búsqueda desde ahora
    'TIME_ZONE': 'America/Argentina/Buenos_Aires'  # Zona horaria para las consultas
}

# Palabras clave para expansión de consultas
QUERY_EXPANSIONS = {
    'inscripcion': ['inscribir', 'anotarse', 'anotar', 'registrar', 'inscripto'],
    'constancia': ['certificado', 'comprobante', 'papel', 'documento'],
    'regular': ['regularidad', 'condición', 'estado', 'situación'],
    'final': ['examen', 'evaluación', 'rendir', 'dar'],
    'recursada': ['recursar', 'volver a cursar', 'segunda vez'],
    'correlativa': ['correlatividad', 'requisito', 'necesito', 'puedo cursar'],
    'baja': ['dar de baja', 'abandonar', 'dejar', 'salir'],
    # Nuevas expansiones
    'denuncia': ['denuncia', 'queja', 'reclamo', 'reportar', 'irregularidad', 'problema', 'presentar', 'acusar'],
    'procedimiento': ['procedimiento', 'proceso', 'pasos', 'cómo', 'manera', 'forma', 'metodología', 'trámite'],
    'sancion': ['sanción', 'sanciones', 'castigo', 'penalidad', 'disciplina', 'apercibimiento', 'suspensión'],
    'sumario': ['sumario', 'investigación', 'proceso disciplinario', 'expediente'],
    'readmision': ['readmisión', 'readmitir', 'volver', 'reincorporación', 'reintegro'],
    'aprobacion': ['aprobar', 'aprobación', 'pasar materias', 'materias aprobadas', 'requisitos'],
    'suspension': ['suspensión', 'suspender', 'interrumpir', 'detener estudios', 'temporalmente']
}

# CONFIGURACIÓN PARA GOOGLE SHEETS (CURSOS)
# El usuario debe asegurarse que la hoja principal se llama 'MARZO' o cambiarlo aquí.
SHEET_QUERY_CONFIG = {
    'sheet_name': os.getenv('GOOGLE_SHEETS_DEFAULT_SHEET_NAME', 'MARZO'),
    'range': 'A:E',
    'header_row': 1,
    'activity_col_name': 'NOMBRE DE ACTIVIDAD',
    'form_col_name': 'FORMULARIO',
    'date_col_name': 'FECHA'
}

# Palabras clave para detectar consultas sobre los cursos
SHEET_COURSE_KEYWORDS = [
    'curso', 'cursos', 'actividad',
    'horario de curso', 'formulario de curso', 'inscripción a curso',
    'suturas', 'rcp', 'primeros auxilios', 'vacunación',
    'link para anotarme', 'anotarme al curso de'
] 