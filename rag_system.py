import os
import uuid
import random
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from unidecode import unidecode

from config.settings import logger, PRIMARY_MODEL, FALLBACK_MODEL, EMBEDDING_MODEL, MAX_OUTPUT_TOKENS, API_TIMEOUT, RAG_NUM_CHUNKS, SIMILARITY_THRESHOLD, EMBEDDINGS_DIR, GOOGLE_API_KEY, CURSOS_SPREADSHEET_ID
from config.constants import INTENT_EXAMPLES, GREETING_WORDS, information_emojis, greeting_emojis, warning_emojis, success_emojis, SHEET_COURSE_KEYWORDS, CALENDAR_INTENT_MAPPING, CALENDAR_MESSAGES
from models.openai_model import OpenAIModel, OpenAIEmbedding
from storage.vector_store import FAISSVectorStore
from utils.date_utils import DateUtils
from handlers.intent_handler import normalize_intent_examples, get_query_intent
from handlers.courses_handler import handle_sheet_course_query
from handlers.calendar_handler import get_calendar_events
from handlers.faqs_handler import handle_faq_query
from services.calendar_service import CalendarService
from services.sheets_service import SheetsService

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.embeddings_dir = Path(EMBEDDINGS_DIR)
        self.user_history = {}
        self.max_history_length = 5
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("Se requiere OPENAI_API_KEY para usar el sistema")
        self.primary_model_name = PRIMARY_MODEL
        self.fallback_model_name = FALLBACK_MODEL
        self.embedding_model_name = EMBEDDING_MODEL
        self.max_output_tokens = MAX_OUTPUT_TOKENS
        self.api_timeout = API_TIMEOUT
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.normalized_intent_examples = normalize_intent_examples(INTENT_EXAMPLES)
        logger.info("Ejemplos de intenciones normalizados para mejorar la clasificación")
        self.model = OpenAIModel(
            model_name=self.primary_model_name,
            api_key=self.openai_api_key,
            timeout=self.api_timeout,
            max_output_tokens=self.max_output_tokens
        )
        self.embedding_model = OpenAIEmbedding(
            model_name=self.embedding_model_name,
            api_key=self.openai_api_key,
            timeout=self.api_timeout
        )
        self.vector_store = FAISSVectorStore(
            str(self.embeddings_dir / 'faiss_index.bin'),
            str(self.embeddings_dir / 'metadata.csv')
        )
        self.date_utils = DateUtils()
        try:
            self.calendar_service = CalendarService()
            logger.info("Servicio de calendario inicializado correctamente")
        except Exception as e:
            logger.warning(f"No se pudo inicializar el servicio de calendario: {str(e)}")
            self.calendar_service = None
        try:
            self.sheets_service = SheetsService(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
            self.CURSOS_SPREADSHEET_ID = CURSOS_SPREADSHEET_ID
        except Exception as e:
            logger.warning(f"No se pudo inicializar el servicio de Google Sheets: {str(e)}")
            self.sheets_service = None

    def normalize_query(self, query: str) -> str:
        """Normaliza la consulta eliminando tildes, signos de puntuación y normalizando espacios."""
        normalized = query.lower().strip()
        normalized = unidecode(normalized)  # Eliminar tildes
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Eliminar signos de puntuación
        normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalizar espacios
        return normalized

    def process_query(self, query: str, user_id: str = None, user_name: str = None) -> Dict[str, Any]:
        if user_id is None:
            user_id = str(uuid.uuid4())
        query_original = query
        query_lower = query.lower().strip()
        
        # Saludo simple
        if len(query.split()) <= 2 and any(saludo in query_lower for saludo in GREETING_WORDS):
            return {
                "query": query_original,
                "response": f"{random.choice(greeting_emojis)} ¡Hola{(' ' + user_name) if user_name else ''}! Soy DrCecim. ¿En qué puedo ayudarte hoy sobre la Facultad de Medicina UBA?",
                "query_type": "saludo",
                "relevant_chunks": [], "sources": []
            }
        
        # Determinar intención - primero con keywords para evitar llamadas innecesarias a la API
        # Pasamos None como embedding_model para usar solo keywords en primera instancia
        intent, confidence = get_query_intent(query_original, None, self.normalized_intent_examples)
        
        # Si no se detectó por keywords o la confianza es baja, usar enfoque basado en palabras clave específicas
        if intent == 'desconocido' or confidence < 0.6:
            # Revisamos si es una pregunta de cortesía común
            if any(keyword in query_lower for keyword in ['como estas', 'cómo estás', 'como te va', 'todo bien']):
                intent = 'cortesia'
                confidence = 0.8
                logger.info(f"Intención detectada por patrones específicos: {intent} (confianza: {confidence:.2f})")
            # Revisamos si pregunta por la identidad
            elif any(keyword in query_lower for keyword in ['como te llamas', 'cómo te llamás', 'tu nombre', 'quien eres']):
                intent = 'identidad'
                confidence = 0.8
                logger.info(f"Intención detectada por patrones específicos: {intent} (confianza: {confidence:.2f})")
            else:
                logger.info(f"Usando enfoque RAG estándar para la consulta")
        else:
            logger.info(f"Intención detectada por keywords: {intent} (confianza: {confidence:.2f})")
        
        # Responder según la intención detectada
        if intent == 'pregunta_capacidades' and confidence > 0.5:
            emoji = random.choice(information_emojis)
            return {
                "query": query_original,
                "response": f"{emoji} Puedo ayudarte con información sobre:\n- Trámites administrativos y procesos académicos de la facultad\n- Fechas de exámenes e inscripciones\n- Requisitos de cursada y regularidad\n- Información sobre cursos y programas\n- Calendario académico y eventos importantes\n\nSi necesitas información específica, ¡pregúntame!",
                "query_type": "pregunta_capacidades",
                "relevant_chunks": [], "sources": []
            }
            
        elif intent == 'identidad' and confidence > 0.5:
            emoji = random.choice(greeting_emojis)
            return {
                "query": query_original,
                "response": f"{emoji} Me llamo DrCecim y soy un asistente virtual especializado de la Facultad de Medicina UBA. Estoy aquí para ayudarte con información sobre trámites, procedimientos administrativos y consultas académicas.",
                "query_type": "identidad",
                "relevant_chunks": [], "sources": []
            }
            
        elif intent == 'pregunta_nombre' and confidence > 0.5:
            emoji = random.choice(greeting_emojis)
            if user_name:
                return {
                    "query": query_original,
                    "response": f"{emoji} Tu nombre es {user_name}. Es un placer ayudarte con información sobre la Facultad de Medicina UBA.",
                    "query_type": "pregunta_nombre",
                    "relevant_chunks": [], "sources": []
                }
            else:
                return {
                    "query": query_original,
                    "response": f"{emoji} No tengo registrado tu nombre en este momento. Si quieres, puedes presentarte y lo recordaré para futuras consultas.",
                    "query_type": "pregunta_nombre",
                    "relevant_chunks": [], "sources": []
                }
                
        elif intent == 'cortesia' and confidence > 0.5:
            emoji = random.choice(greeting_emojis)
            return {
                "query": query_original,
                "response": f"{emoji} ¡Estoy muy bien, gracias por preguntar! ¿En qué puedo ayudarte hoy con información sobre la Facultad de Medicina?",
                "query_type": "cortesia",
                "relevant_chunks": [], "sources": []
            }
            
        elif intent == 'agradecimiento' and confidence > 0.5:
            emoji = random.choice(success_emojis)
            return {
                "query": query_original,
                "response": f"{emoji} ¡De nada! Estoy para ayudarte. Si necesitas algo más, no dudes en preguntar.",
                "query_type": "agradecimiento",
                "relevant_chunks": [], "sources": []
            }
            
        elif intent == 'consulta_medica' and confidence > 0.5:
            emoji = random.choice(warning_emojis)
            return {
                "query": query_original,
                "response": f"{emoji} Lo siento, no puedo responder consultas médicas. Te recomiendo dirigirte a un profesional de la salud o al servicio médico de la facultad para recibir atención adecuada.",
                "query_type": "consulta_medica",
                "relevant_chunks": [], "sources": []
            }
            
        # Para consultas con baja confianza, no preguntar directamente sino tratar de responder
        # con el sistema RAG normal, dejando que el modelo interprete directamente
        
        # Preguntas Frecuentes (FAQs)
        faq_response = handle_faq_query(query_original)
        if faq_response:
            logger.info(f"Consulta respondida desde el manejador de FAQs")
            self.update_user_history(user_id, query_original, faq_response)
            return {
                "query": query_original, 
                "response": faq_response,
                "query_type": "faq",
                "relevant_chunks": [], 
                "sources": ["Preguntas Frecuentes"]
            }
        
        # Cursos (Google Sheets)
        if self.sheets_service and any(keyword in query_lower for keyword in SHEET_COURSE_KEYWORDS):
            sheet_course_response = handle_sheet_course_query(
                query_original, self.sheets_service, self.CURSOS_SPREADSHEET_ID, self.date_utils
            )
            if sheet_course_response:
                self.update_user_history(user_id, query_original, sheet_course_response)
                return {
                    "query": query_original, "response": sheet_course_response,
                    "query_type": "info_cursos",
                    "relevant_chunks": [], "sources": [f"Google Sheet Cursos"]
                }
        # Calendario
        calendar_intent = None
        for intent, config in CALENDAR_INTENT_MAPPING.items():
            if any(keyword in query_lower for keyword in config['keywords']):
                calendar_intent = intent
                break
        if calendar_intent:
            calendar_response = get_calendar_events(self.calendar_service, calendar_intent)
            self.update_user_history(user_id, query_original, calendar_response)
            return {
                "query": query_original,
                "response": calendar_response,
                "query_type": f"calendario_{calendar_intent}",
                "relevant_chunks": [],
                "sources": ["Calendario Académico"]
            }
        # RAG clásico
        relevant_chunks = self.retrieve_relevant_chunks(query_original, k=RAG_NUM_CHUNKS)
        sources = [chunk.get('filename', '') for chunk in relevant_chunks if chunk.get('filename')]
        context = '\n\n'.join(chunk.get('text', '') for chunk in relevant_chunks)
        if not context.strip():
            emoji = random.choice(information_emojis)
            return {
                "query": query_original,
                "response": f"{emoji} Lo siento, no encontré información específica sobre esta consulta en mis documentos. Te sugiero escribir a alumnos@fmed.uba.ar para obtener la información precisa que necesitas.",
                "relevant_chunks": [],
                "sources": [],
                "query_type": "sin_información"
            }
        response = self.generate_response(query_original, context, sources)
        self.update_user_history(user_id, query_original, response)
        return {
            "query": query_original,
            "response": response,
            "relevant_chunks": relevant_chunks,
            "sources": sources,
            "query_type": "consulta_general"
        }

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        query = query.strip()
        if not query:
            return []
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.vector_store.search(query_embedding, k=k)
        return results

    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        emoji = random.choice(information_emojis)
        sources_text = f"\nFUENTES CONSULTADAS:\n{', '.join(sources)}" if sources else ""
        prompt = f"""
Sos DrCecim, un asistente virtual especializado de la Facultad de Medicina UBA. Tu tarea es proporcionar respuestas sobre administración y trámites de la facultad y deben ser breves, precisas y útiles.

INFORMACIÓN RELEVANTE:
{context}
{sources_text}

CONSULTA ACTUAL: {query}

RESPONDE SIGUIENDO ESTAS REGLAS:
1. Sé muy conciso y directo
2. Usa la información de los documentos oficiales primero
3. Si hay documentos específicos, cita naturalmente su origen ("Según el reglamento...")
4. NO uses NUNCA formato Markdown (como asteriscos para negrita o cursiva) ya que esto no se procesa correctamente en WhatsApp
5. Para enfatizar texto, usa MAYÚSCULAS, comillas o asteriscos
6. Usa viñetas con guiones (-) cuando sea útil para mayor claridad
7. Si la información está incompleta, sugiere contactar a @cecim.nemed por instagram
8. No hagas preguntas adicionales
"""
        try:
            response = self.model.generate(prompt)
        except Exception:
            return f"{emoji} Lo siento, hubo un error al generar la respuesta. Por favor, intenta de nuevo o contacta a @cecim.nemed por instagram."
        response = re.sub(r'\*\*(.+?)\*\*', r'\1', response)
        response = re.sub(r'\*(.+?)\*', r'\1', response)
        response = re.sub(r'\_\_(.+?)\_\_', r'\1', response)
        response = re.sub(r'\_(.+?)\_', r'\1', response)
        if not re.match(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]', response.strip()[0:1]):
            response = f"{emoji} {response}"
        return response

    def update_user_history(self, user_id: str, query: str, response: str):
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append({
            "query": query,
            "response": response,
            "timestamp": time.time()
        })
        if len(self.user_history[user_id]) > self.max_history_length:
            self.user_history[user_id] = self.user_history[user_id][-self.max_history_length:] 