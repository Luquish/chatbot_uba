"""
Manejador de intenciones para clasificar las consultas de los usuarios.
"""
import logging
import re
import numpy as np
from typing import Tuple, Dict, List
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity

from config.constants import INTENT_EXAMPLES
from models.openai_model import OpenAIEmbedding

logger = logging.getLogger(__name__)


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
            norm_example = example.lower().strip()
            norm_example = unidecode(norm_example)  # Eliminar tildes
            norm_example = re.sub(r'[^\w\s]', '', norm_example)  # Eliminar signos de puntuación
            norm_example = re.sub(r'\s+', ' ', norm_example).strip()  # Normalizar espacios
            norm_examples.append(norm_example)
            
        normalized_examples[intent] = {
            'examples': norm_examples,
            'context': data['context']
        }
        
    return normalized_examples


def get_query_intent(query: str, embedding_model: OpenAIEmbedding, normalized_intent_examples: Dict = None) -> Tuple[str, float]:
    """
    Determina la intención de la consulta usando similitud semántica.
    
    Args:
        query (str): Consulta del usuario
        embedding_model (OpenAIEmbedding): Modelo de embeddings
        normalized_intent_examples (Dict): Ejemplos normalizados (opcional)
        
    Returns:
        Tuple[str, float]: Intención detectada y nivel de confianza
    """
    # Si no se proporcionan ejemplos normalizados, normalizarlos
    if normalized_intent_examples is None:
        normalized_intent_examples = normalize_intent_examples(INTENT_EXAMPLES)
    
    # Normalización del texto
    query_original = query
    query = query.lower().strip()
    query = unidecode(query)  # Eliminar tildes
    query = re.sub(r'[^\w\s]', '', query)  # Eliminar signos de puntuación
    query = re.sub(r'\s+', ' ', query).strip()  # Normalizar espacios
    
    if query_original != query:
        logger.info(f"Consulta normalizada: '{query_original}' → '{query}'")
        
    query_embedding = embedding_model.encode([query])[0]
    
    max_similarity = -1
    best_intent = 'desconocido'
    
    # Usar ejemplos normalizados
    for intent, data in normalized_intent_examples.items():
        examples = data['examples']
        example_embeddings = embedding_model.encode(examples)
        similarities = cosine_similarity([query_embedding], example_embeddings)[0]
        avg_similarity = np.mean(similarities)
        
        # Para debugging
        logger.debug(f"Intención: {intent}, similitud: {avg_similarity:.2f}")
        
        if avg_similarity > max_similarity:
            max_similarity = avg_similarity
            best_intent = intent
    
    return best_intent, max_similarity


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