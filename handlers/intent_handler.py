"""
Manejador de intenciones para clasificar las consultas de los usuarios.
"""
import logging
import re
from typing import Tuple, Dict, List, Optional
from unidecode import unidecode

from config.constants import INTENT_EXAMPLES
from models.openai_model import OpenAIEmbedding, OpenAIModel

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normaliza un texto para comparación.
    
    Args:
        text (str): Texto a normalizar
        
    Returns:
        str: Texto normalizado
    """
    normalized = text.lower().strip()
    normalized = unidecode(normalized)  # Eliminar tildes
    normalized = re.sub(r'[^\w\s]', '', normalized)  # Eliminar signos de puntuación
    normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalizar espacios
    return normalized


def normalize_intent_examples(intent_examples: Dict) -> Dict:
    """
    Normaliza los ejemplos de intenciones para hacer comparaciones más robustas.
    
    Args:
        intent_examples (Dict): Diccionario con ejemplos de intenciones
        
    Returns:
        Dict: Ejemplos normalizados
    """
    normalized_examples = {}
    
    for intent, data in intent_examples.items():
        examples = data['examples']
        norm_examples = []
        
        for example in examples:
            # Aplicar la misma normalización que a las consultas
            norm_examples.append(normalize_text(example))
            
        normalized_examples[intent] = {
            'examples': norm_examples,
            'context': data['context']
        }
        
    return normalized_examples


def get_intent_by_keywords(query: str, normalized_intent_examples: Dict) -> Optional[Tuple[str, float]]:
    """
    Detecta la intención basada en palabras clave para casos comunes.
    Más eficiente que usar embeddings para consultas simples.
    
    Args:
        query (str): Consulta normalizada
        normalized_intent_examples (Dict): Ejemplos normalizados
        
    Returns:
        Optional[Tuple[str, float]]: (intent, confianza) si se detecta, None si no
    """
    # Mapeo de palabras clave a intenciones con alta confianza
    keyword_patterns = {
        # Saludos
        'saludo': ['hola', 'buenas', 'buenos dias', 'buen dia', 'que tal', 'saludos'],
        
        # Identidad
        'identidad': ['quien sos', 'quien eres', 'como te llamas', 'cual es tu nombre', 'que sos vos', 
                      'tu nombre', 'sos un bot', 'eres un bot', 'sos una persona', 'como se llama'],
        
        # Pregunta sobre capacidades
        'pregunta_capacidades': ['que podes hacer', 'que sabes hacer', 'que te puedo preguntar', 
                                'que puedo preguntarte', 'en que me ayudas', 'para que servis'],
        
        # Cortesía
        'cortesia': ['como estas', 'como va', 'como te va', 'todo bien', 'como andas'],
        
        # Agradecimientos
        'agradecimiento': ['gracias', 'muchas gracias', 'te agradezco', 'genial', 'perfecto', 'ok', 'okey', 'listo']
    }
    
    # Normalizar la consulta
    query_norm = normalize_text(query)
    
    # Primero buscar coincidencias exactas en ejemplos normalizados
    for intent, data in normalized_intent_examples.items():
        if query_norm in data['examples']:
            return intent, 0.95
    
    # Luego buscar palabras clave para intenciones comunes
    for intent, keywords in keyword_patterns.items():
        for keyword in keywords:
            # Coincidencia completa con palabra clave (alta confianza)
            if keyword == query_norm:
                return intent, 0.9
            
            # La consulta contiene la palabra clave (menor confianza)
            if keyword in query_norm and len(query_norm.split()) <= 5:
                return intent, 0.75
    
    # No se encontró coincidencia por palabras clave
    return None


def get_query_intent(query: str, embedding_model: Optional[OpenAIEmbedding] = None, 
                    normalized_intent_examples: Dict = None) -> Tuple[str, float]:
    """
    Determina la intención de la consulta usando palabras clave primero y, si es necesario, 
    similitud semántica con embeddings.
    
    Args:
        query (str): Consulta del usuario
        embedding_model (OpenAIEmbedding): Modelo de embeddings (opcional)
        normalized_intent_examples (Dict): Ejemplos normalizados (opcional)
        
    Returns:
        Tuple[str, float]: Intención detectada y nivel de confianza
    """
    # Si no se proporcionan ejemplos normalizados, normalizarlos
    if normalized_intent_examples is None:
        normalized_intent_examples = normalize_intent_examples(INTENT_EXAMPLES)
    
    # Guardar consulta original para logging
    query_original = query
    
    # Normalización del texto
    query_norm = normalize_text(query)
    
    if query_original != query_norm:
        logger.info(f"Consulta normalizada: '{query_original}' → '{query_norm}'")
    
    # Primero intentar detección basada en palabras clave (más eficiente)
    keyword_result = get_intent_by_keywords(query_norm, normalized_intent_examples)
    if keyword_result:
        intent, confidence = keyword_result
        logger.info(f"Intención detectada por palabras clave: '{intent}' (confianza: {confidence:.2f})")
        return intent, confidence
    
    # Si no hay modelo de embeddings o se omite intencionalmente, devolver valor por defecto
    if embedding_model is None:
        logger.info("No se proporcionó modelo de embeddings, retornando intención desconocida")
        return 'desconocido', 0.0
    
    # Si no se pudo determinar por palabras clave y se cuenta con modelo, 
    # usar embedding para consultas más complejas
    logger.info("Usando modelo de embeddings para detectar intención")
    
    try:
        # Analizar solo una muestra de ejemplos por intención para reducir carga
        sample_examples = []
        sample_intents = []
        
        for intent, data in normalized_intent_examples.items():
            # Tomar solo hasta 3 ejemplos por intención para reducir llamadas a la API
            examples_to_use = data['examples'][:3]
            sample_examples.extend(examples_to_use)
            sample_intents.extend([intent] * len(examples_to_use))
        
        # Solo una llamada a la API para todos los ejemplos
        query_embedding = embedding_model.encode([query_norm])[0]
        all_embeddings = embedding_model.encode(sample_examples)
        
        # Encontrar el ejemplo más similar
        max_similarity = -1
        best_intent = 'desconocido'
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        
        best_idx = similarities.argmax()
        max_similarity = similarities[best_idx]
        best_intent = sample_intents[best_idx]
        
        logger.info(f"Intención detectada por embeddings: '{best_intent}' (similitud: {max_similarity:.2f})")
        
        return best_intent, float(max_similarity)
        
    except Exception as e:
        logger.error(f"Error al usar embeddings: {str(e)}")
        return 'desconocido', 0.0


def interpret_ambiguous_query(model: OpenAIModel, query: str) -> Tuple[str, float]:
    """
    Usa el modelo de lenguaje para interpretar una consulta ambigua y determinar la intención.
    
    Args:
        model (OpenAIModel): Modelo de lenguaje
        query (str): Consulta del usuario
        
    Returns:
        Tuple[str, float]: Intención interpretada y nivel de confianza
    """
    prompt = f"""
Eres un asistente especializado en detectar la intención de consultas de usuarios para un chatbot universitario.
Debes determinar qué intención corresponde mejor a la siguiente consulta del usuario:

Consulta: "{query}"

Estas son las posibles intenciones:
- saludo: El usuario está saludando o iniciando la conversación
- identidad: El usuario pregunta por el nombre o identidad del bot
- pregunta_capacidades: El usuario pregunta qué puede hacer el bot
- cortesia: El usuario pregunta cómo está el bot, es una interacción social
- agradecimiento: El usuario está agradeciendo o mostrando conformidad
- consulta_administrativa: El usuario pregunta sobre trámites administrativos
- consulta_academica: El usuario pregunta sobre aspectos académicos
- consulta_medica: El usuario hace una consulta médica
- desconocido: La intención no corresponde a ninguna categoría conocida

Responde sólo con la intención que corresponda sin explicaciones adicionales. Si no estás seguro, responde "desconocido".
"""
    try:
        response = model.generate(prompt).strip().lower()
        # Si la respuesta corresponde a una intención válida, asignar confianza media
        valid_intents = ["saludo", "identidad", "pregunta_capacidades", "cortesia", 
                         "agradecimiento", "consulta_administrativa", "consulta_academica", 
                         "consulta_medica", "desconocido"]
        
        if response in valid_intents:
            return response, 0.7
        else:
            logger.warning(f"Respuesta no válida del modelo: {response}")
            return "desconocido", 0.0
    except Exception as e:
        logger.error(f"Error al interpretar consulta con modelo: {str(e)}")
        return "desconocido", 0.0


def generate_conversational_response(model, query: str, intent: str, user_name: str = None) -> str:
    """
    Genera una respuesta conversacional basada en la intención detectada.
    
    Args:
        model: Modelo de lenguaje a utilizar
        query (str): Consulta del usuario
        intent (str): Intención detectada
        user_name (str): Nombre del usuario (opcional)
        
    Returns:
        str: Respuesta generada
    """
    context = INTENT_EXAMPLES[intent]['context'] if intent in INTENT_EXAMPLES else "Consulta general"
    
    # Determinar si es el primer mensaje o un saludo del usuario
    is_greeting = intent == 'saludo'
    is_courtesy = intent == 'cortesia'
    is_acknowledgment = intent == 'agradecimiento'
    is_capabilities = intent == 'pregunta_capacidades'
    
    # Personalizar el prompt según si tenemos el nombre del usuario
    user_context = f"El usuario se llama {user_name}. " if user_name else ""
    
    prompt = f"""
Como DrCecim, un asistente virtual de la Facultad de Medicina de la UBA:
- Usa un tono amigable y profesional
- Mantén las respuestas breves y directas
- No hagas preguntas adicionales
- Solo saluda si el usuario está saludando por primera vez
- Si conoces el nombre del usuario, úsalo de manera natural sin forzarlo

{user_context}
Contexto de la consulta: {context}
Consulta del usuario: {query}

Instrucciones específicas:
- Si es un saludo: {"Saluda usando el nombre del usuario si está disponible y menciona que puedes ayudar con trámites y consultas" if is_greeting else "Responde directamente sin saludar"}
- Si es una pregunta de cortesía: {"Responde amablemente mencionando el nombre si está disponible, pero sin volver a presentarte" if is_courtesy else "Responde directamente"}
- Si preguntan sobre tus capacidades: Explica que ayudas con trámites administrativos y consultas académicas
- Si es una consulta médica: Explica amablemente que no puedes responder consultas médicas
- Si preguntan tu identidad: Explica que eres un asistente virtual de la facultad, sin saludar nuevamente
"""

    return model.generate(prompt) 