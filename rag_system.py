import os
import uuid
import random
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from config.settings import logger, PRIMARY_MODEL, FALLBACK_MODEL, EMBEDDING_MODEL, MAX_OUTPUT_TOKENS, API_TIMEOUT, RAG_NUM_CHUNKS, SIMILARITY_THRESHOLD, EMBEDDINGS_DIR, GOOGLE_API_KEY, CURSOS_SPREADSHEET_ID
from config.constants import INTENT_EXAMPLES, GREETING_WORDS, information_emojis, greeting_emojis, warning_emojis, SHEET_COURSE_KEYWORDS, CALENDAR_INTENT_MAPPING, CALENDAR_MESSAGES
from models.openai_model import OpenAIModel, OpenAIEmbedding
from storage.vector_store import FAISSVectorStore
from utils.date_utils import DateUtils
from handlers.intent_handler import normalize_intent_examples, get_query_intent, generate_conversational_response
from handlers.courses_handler import handle_sheet_course_query
from handlers.calendar_handler import get_calendar_events
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
            
        # Detectar preguntas sobre capacidades usando palabras clave
        capacidades_keywords = ["que podes hacer", "que sabes hacer", "que te puedo preguntar", "que puedo preguntarte", 
                                "que tipos de dudas", "que consultas", "en que me podes ayudar", "para que servis",
                                "que te puedo consultar", "que puedo consultarte"]
        if any(keyword in query_lower for keyword in capacidades_keywords):
            emoji = random.choice(information_emojis)
            return {
                "query": query_original,
                "response": f"{emoji} Puedo ayudarte con información sobre:\n- Trámites administrativos y procesos académicos de la facultad\n- Fechas de exámenes e inscripciones\n- Requisitos de cursada y regularidad\n- Información sobre cursos y programas\n- Calendario académico y eventos importantes\n\nSi necesitas información específica, ¡pregúntame!",
                "query_type": "pregunta_capacidades",
                "relevant_chunks": [], "sources": []
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