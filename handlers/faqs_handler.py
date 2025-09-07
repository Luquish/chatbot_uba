"""
Manejador para consultas relacionadas con preguntas frecuentes.
"""
import logging
import random
import re
from typing import Dict, List, Any, Optional

from config.constants import information_emojis
from utils.text_utils import normalize_text

logger = logging.getLogger(__name__)

# Lista de temas de FAQs para detección
FAQ_KEYWORDS = {
    'constancia_alumno_regular': [
        'constancia', 'certificado', 'certificación', 'comprobante', 'tramitar regularidad',
        'obtener constancia', 'solicitar constancia', 'tramitar certificado', 'solicitar certificado',
        'ventanilla', 'imprimir constancia'
    ],
    'baja_materia': [
        'baja', 'dar de baja', 'abandonar materia', 'dejar materia', 'anular materia', 'cancelar materia'
    ],
    'anulacion_final': [
        'anular final', 'cancelar final', 'anulación final', 'cancelación final', 'dar de baja final'
    ],
    'problemas_inscripcion': [
        'no pude inscribirme', 'problema inscripción', 'no me asignaron', 'no asignaron materia', 
        'error inscripción', 'falla inscripción', 'no me dejó inscribir'
    ],
    'reincorporacion': [
        'reincorporación', 'reincorporarme', 'readmisión', 'volver a la carrera', 'retomar estudios'
    ],
    'recursada': [
        'recursada', 'recursar', 'segunda vez', 'volver a cursar', 'segunda cursada'
    ],
    'tercera_cursada': [
        'tercera cursada', 'tercera vez', 'cursar tercera', '3ra cursada', '3era cursada'
    ],
    'cuarta_cursada': [
        'cuarta cursada', 'cuarta vez', 'más de tres veces', '4ta cursada', '4ta vez'
    ],
    'prorroga': [
        'prórroga', 'prorroga', 'extender', 'extensión materia', 'ampliar plazo'
    ]
}

# Palabras clave que indican consulta sobre normativas de regularidad (no sobre el trámite)
NORMAS_REGULARIDAD_KEYWORDS = [
    'condiciones de regularidad', 'condiciones para regularidad', 'mantener regularidad',
    'requisitos regularidad', 'perder regularidad', 'porcentaje aplazos', 'cuántas materias',
    'tiempo completar carrera', 'materias aprobar', 'condición de alumno', 'normativa regularidad',
    'régimen de regularidad', 'reglamento regularidad'
]

# Preguntas específicas conocidas que nos pueden ayudar a identificar la intención
FAQ_QUESTIONS = {
    'constancia_alumno_regular': [
        '¿cómo obtengo la constancia de alumno regular?',
        '¿dónde tramito la constancia de alumno regular?',
        '¿cómo hago para obtener un certificado de regularidad?'
    ],
    'baja_materia': [
        '¿cómo doy de baja una materia?',
        '¿hasta cuándo puedo dar de baja una materia?',
        '¿cuál es el procedimiento para dejar una materia?'
    ],
    'anulacion_final': [
        '¿cómo anulo la inscripción a un final?',
        '¿puedo cancelar un final?',
        '¿dónde me doy de baja de un examen final?'
    ],
    'problemas_inscripcion': [
        '¿qué hago si no me puedo inscribir?',
        '¿qué pasa si no me asignaron una materia?',
        '¿a dónde voy si no logré inscribirme?'
    ],
    'reincorporacion': [
        '¿cómo me reincorporo a la carrera?',
        '¿cuál es el procedimiento de reincorporación?',
        '¿cómo hago para volver a la facultad si dejé un tiempo?'
    ],
    'recursada': [
        '¿cómo me inscribo para recursar?',
        '¿qué tengo que hacer para cursar una materia por segunda vez?',
        '¿hay que pagar para recursar?'
    ],
    'tercera_cursada': [
        '¿cómo inscribo para una tercera cursada?',
        '¿qué necesito para cursar una materia por tercera vez?',
        '¿hay que pagar para la tercera cursada?'
    ],
    'cuarta_cursada': [
        '¿se puede cursar una materia más de tres veces?',
        '¿cómo inscribo para una cuarta cursada?',
        '¿qué trámite hago para la cuarta cursada?'
    ],
    'prorroga': [
        '¿cómo solicito prórroga de una materia?',
        '¿qué es una prórroga de materia?',
        '¿dónde pido una prórroga?'
    ]
}

# Respuestas asociadas a cada tema de FAQ
FAQ_RESPONSES = {
    'constancia_alumno_regular': 
        "Para tramitar la constancia de alumno regular, debes:\n"
        "1. Ingresar al Sitio de Inscripciones con tu DNI y contraseña\n"
        "2. Seleccionar \"Constancia de alumno regular\" en el inicio de trámites\n"
        "3. Imprimir la constancia\n"
        "4. Presentarte con tu Libreta Universitaria o DNI y el formulario impreso en la ventanilla del Ciclo Biomédico",
    
    'baja_materia':
        "Para dar de baja una materia:\n"
        "- Debes hacerlo hasta 2 semanas antes del primer parcial o hasta el 25% de la cursada en materias sin parcial\n"
        "- En el Sitio de Inscripciones, selecciona \"Baja de asignatura\"\n"
        "- Imprime el certificado de baja\n"
        "- Una vez finalizado, el estado será \"Resuelto Positivamente\" y no deberás acudir a la Dirección de Alumnos",
    
    'anulacion_final':
        "Para anular la inscripción a un final, debes acudir a la ventanilla del Ciclo Biomédico presentando el número de constancia generado durante el trámite de inscripción.",
    
    'problemas_inscripcion':
        "Si no lograste inscribirte o no te asignaron una materia, debes dirigirte a la cátedra o departamento correspondiente y solicitar la inclusión en lista, presentando tu Libreta Universitaria o DNI.",
    
    'reincorporacion':
        "Para reincorporarte a la carrera:\n"
        "- Ingresa al Sitio de Inscripciones y selecciona \"Reincorporación a la carrera\"\n"
        "- Si es tu 1ª reincorporación: El trámite es automático sin necesidad de ir a ventanilla\n"
        "- Si ya fuiste reincorporado antes: Debes imprimir el trámite (2 hojas) y presentarlo en la ventanilla del Ciclo Biomédico para evaluación por la Comisión de Readmisión",
    
    'recursada':
        "Para solicitar una recursada:\n"
        "- Genera el trámite en el Sitio de Inscripciones seleccionando \"Recursada\"\n"
        "- Si en el sistema apareces como dado DE BAJA en la cursada anterior: Solo genera el trámite sin pagar\n"
        "- Si no apareces dado DE BAJA: Deberás 1) Realizar el trámite, 2) Generar e imprimir el talón de pago, 3) Pagar en Tesorería, 4) Presentar comprobante en los buzones del Ciclo Biomédico",
    
    'tercera_cursada':
        "Para solicitar la tercera cursada:\n"
        "- En el Sitio de Inscripciones, selecciona \"3º Cursada\"\n"
        "- Imprime la constancia y el certificado\n"
        "- Si figuras como dado DE BAJA en las dos cursadas anteriores: Te inscribes sin abonar arancel\n"
        "- Si no: Debes 1) Realizar el trámite, 2) Generar e imprimir el talón de pago, 3) Pagar en Tesorería, 4) Presentar comprobante en el buzón del Ciclo Biomédico",
    
    'cuarta_cursada':
        "Para la cuarta cursada o más:\n"
        "- En el Sitio de Inscripciones, selecciona \"4º Cursada o más\"\n"
        "- Imprime la constancia y el certificado\n"
        "- Debes presentarte con tu Libreta Universitaria y las constancias impresas en la ventanilla del Ciclo Biomédico y acudir a la Dirección de Alumnos",
    
    'prorroga':
        "Para solicitar prórroga de una asignatura:\n"
        "- En el Sitio de Inscripciones, selecciona \"Prórroga de asignatura\"\n"
        "- Imprime la constancia\n"
        "- Para 1ª o 2ª prórroga: El trámite se resuelve positivamente automáticamente\n"
        "- Para 3ª prórroga o superior: Debes presentar la constancia y tu Libreta Universitaria en la ventanilla del Ciclo Biomédico"
}

def get_faq_intent(query: str) -> Optional[Dict[str, Any]]:
    """
    Determina si una consulta está relacionada con alguna FAQ y retorna la información.
    
    Args:
        query (str): Consulta del usuario
        
    Returns:
        Optional[Dict[str, Any]]: Diccionario con información de la FAQ o None si no hay match
    """
    # Normalizar la consulta
    query_norm = normalize_text(query)
    logger.info(f"Consulta normalizada para FAQs: '{query_norm}'")
    
    # Revisar primero si la consulta es sobre normativas de regularidad (no sobre el trámite)
    # En ese caso, no queremos responder con la FAQ de constancia de alumno regular
    for keyword in NORMAS_REGULARIDAD_KEYWORDS:
        keyword_norm = normalize_text(keyword)
        if keyword_norm in query_norm:
            logger.info(f"Detectada consulta sobre normativas de regularidad, evitando respuesta FAQ")
            return None
    
    # Verificar coincidencias exactas con preguntas conocidas
    for intent, questions in FAQ_QUESTIONS.items():
        for question in questions:
            question_norm = normalize_text(question)
            # Si hay una coincidencia de alta confianza con una pregunta conocida
            if query_norm in question_norm or question_norm in query_norm:
                logger.info(f"Coincidencia directa con pregunta FAQ: {intent}")
                return {
                    'intent': intent,
                    'confidence': 0.9,
                    'response': FAQ_RESPONSES[intent]
                }
    
    # Verificar coincidencias basadas en keywords
    best_intent = None
    best_score = 0
    
    for intent, keywords in FAQ_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            keyword_norm = normalize_text(keyword)
            if keyword_norm in query_norm:
                # Incrementar score basado en la especificidad de la keyword
                if len(keyword_norm) > 10:
                    score += 2  # Keyword más específica tiene mayor peso
                else:
                    score += 1
        
        # Caso especial para constancia de alumno regular
        # Reducir la puntuación si solo contiene la palabra "regularidad" sin otras keywords específicas
        if intent == 'constancia_alumno_regular' and score == 1 and 'regularidad' in query_norm:
            if not any(kw in query_norm for kw in ['constancia', 'certificado', 'tramitar', 'solicitar', 'obtener']):
                score = 0
                logger.info("Se detectó 'regularidad' pero parece ser sobre normativas, no sobre constancia")
        
        # Normalizar score según cantidad de keywords
        norm_score = score / (len(keywords) * 0.7)  # Factor de ajuste para no requerir todas las keywords
        
        if norm_score > best_score and norm_score > 0.25:  # Umbral mínimo
            best_score = norm_score
            best_intent = intent
    
    if best_intent:
        # Reducir la confianza para constancia_alumno_regular si la consulta parece ser sobre normativas
        if best_intent == 'constancia_alumno_regular' and any(term in query_norm for term in 
                                                        ['condicion', 'mantener', 'requisito', 'perder']):
            best_score = best_score * 0.5
            logger.info(f"Reduciendo confianza para constancia_alumno_regular: {best_score:.2f}")
            
            # Si la confianza cae por debajo del umbral, no considerarla como coincidencia
            if best_score < 0.25:
                logger.info("Confianza insuficiente después del ajuste, no se considera coincidencia FAQ")
                return None
        
        confidence = min(best_score, 0.85)  # Limitar confianza máxima para coincidencias por keywords
        logger.info(f"Coincidencia por keywords FAQ: {best_intent} (confianza: {confidence:.2f})")
        return {
            'intent': best_intent,
            'confidence': confidence,
            'response': FAQ_RESPONSES[best_intent]
        }
    
    return None


def handle_faq_query(query: str) -> Optional[str]:
    """
    Maneja una consulta relacionada con FAQs.
    
    Args:
        query (str): Consulta del usuario
        
    Returns:
        Optional[str]: Respuesta formateada o None si no se detecta relación con FAQs
    """
    faq_result = get_faq_intent(query)
    
    if faq_result and faq_result['confidence'] > 0.25:
        emoji = random.choice(information_emojis)
        return f"{emoji} {faq_result['response']}"
    
    return None 