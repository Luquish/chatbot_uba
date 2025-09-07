"""
Sistema RAG refactorizado para la Facultad de Medicina UBA.
Versión modularizada con responsabilidades separadas.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

from config.settings import (
    logger, PRIMARY_MODEL, FALLBACK_MODEL, EMBEDDING_MODEL,
    MAX_OUTPUT_TOKENS, API_TIMEOUT, SIMILARITY_THRESHOLD,
    GOOGLE_API_KEY, CURSOS_SPREADSHEET_ID
)
from config.constants import INTENT_EXAMPLES
from models.openai_model import OpenAIModel, OpenAIEmbedding
from storage.vector_store import PostgreSQLVectorStore
from utils.date_utils import DateUtils
from handlers.intent_handler import normalize_intent_examples
from services.calendar_service import CalendarService
from services.sheets_service import SheetsService
from services.router_service import Router

# Importar nuevos módulos
from core.query_processor import QueryProcessor
from core.response_generator import ResponseGenerator
from core.context_retriever import ContextRetriever

# Importar herramientas
from services.tools.calendar_tool import CalendarTool
from services.tools.faq_tool import FaqTool
from services.tools.rag_tool import RagTool
from services.tools.sheets_tool import SheetsTool
from services.tools.horarios_catedra_tool import HorariosCatedraTool
from services.tools.horarios_lic_tec_tool import HorariosLicTecTool
from services.tools.horarios_secretarias_tool import HorariosSecretariasTool
from services.tools.mails_nuevo_espacio_tool import MailsNuevoEspacioTool
from services.tools.conversational_tool import ConversationalTool
from services.tools.hospitales_tool import HospitalesTool

load_dotenv()


class RAGSystem:
    """Sistema RAG principal refactorizado con módulos especializados."""
    
    def __init__(self):
        self.user_history = {}
        self.max_history_length = 5
        
        # Validar y configurar API key
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("Se requiere OPENAI_API_KEY para usar el sistema")
        
        # Configurar modelos y parámetros
        self._setup_models_and_params()
        
        # Inicializar componentes principales
        self._initialize_core_components()
        
        # Inicializar servicios externos
        self._initialize_external_services()
        
        # Configurar módulos especializados
        self._setup_specialized_modules()
        
        logger.info("Sistema RAG inicializado correctamente con arquitectura modular")
    
    def _setup_models_and_params(self):
        """Configura modelos y parámetros del sistema."""
        self.primary_model_name = PRIMARY_MODEL
        self.fallback_model_name = FALLBACK_MODEL
        self.embedding_model_name = EMBEDDING_MODEL
        self.max_output_tokens = MAX_OUTPUT_TOKENS
        self.api_timeout = API_TIMEOUT
        self.similarity_threshold = SIMILARITY_THRESHOLD
        
        # Normalizar ejemplos de intención
        self.normalized_intent_examples = normalize_intent_examples(INTENT_EXAMPLES)
        logger.info("Ejemplos de intenciones normalizados para mejorar la clasificación")
    
    def _initialize_core_components(self):
        """Inicializa componentes principales de OpenAI y vector store."""
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
        
        # Inicializar vector store con PostgreSQL/pgvector
        self.vector_store = PostgreSQLVectorStore(
            threshold=self.similarity_threshold
        )
        
        logger.info("Vector store configurado con PostgreSQL/pgvector")
        logger.info(f"Umbral de similitud: {self.similarity_threshold}")
        
        # Utilidades de fecha
        self.date_utils = DateUtils()
    
    def _initialize_external_services(self):
        """Inicializa servicios externos (Calendar, Sheets)."""
        # Servicio de calendario
        try:
            self.calendar_service = CalendarService()
            logger.info("Servicio de calendario inicializado correctamente")
        except Exception as e:
            logger.warning(f"No se pudo inicializar el servicio de calendario: {str(e)}")
            self.calendar_service = None
        
        # Servicio de Google Sheets
        try:
            self.sheets_service = SheetsService(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
            self.CURSOS_SPREADSHEET_ID = CURSOS_SPREADSHEET_ID
        except Exception as e:
            logger.warning(f"No se pudo inicializar el servicio de Google Sheets: {str(e)}")
            self.sheets_service = None
    
    def _setup_specialized_modules(self):
        """Configura los módulos especializados."""
        # Recuperador de contexto
        self.context_retriever = ContextRetriever(
            self.embedding_model,
            self.vector_store,
            self.similarity_threshold
        )
        
        # Generador de respuestas
        self.response_generator = ResponseGenerator(
            self.model,
            self.fallback_model_name,
            self.openai_api_key,
            self.api_timeout,
            self.max_output_tokens
        )
        
        # Inicializar Router y herramientas
        self.router = Router(tools=[
            ConversationalTool(self.model),
            FaqTool(),
            HospitalesTool(),
            CalendarTool(self.calendar_service),
            SheetsTool(self.sheets_service, self.date_utils),
            HorariosCatedraTool(self.sheets_service),
            HorariosLicTecTool(self.sheets_service),
            HorariosSecretariasTool(self.sheets_service),
            MailsNuevoEspacioTool(self.sheets_service),
            RagTool(self.embedding_model, self.vector_store, self.generate_response)
        ], config_path='config/router.yaml')
        
        # Procesador de consultas
        self.query_processor = QueryProcessor(
            self.router,
            self.sheets_service,
            self.CURSOS_SPREADSHEET_ID,
            self.date_utils,
            self.normalized_intent_examples,
            self.update_user_history
        )
    
    def process_query(self, query: str, user_id: str = None, user_name: str = None) -> Dict[str, Any]:
        """
        Procesa una consulta del usuario usando el procesador modular.
        
        Args:
            query: La consulta del usuario
            user_id: ID único del usuario
            user_name: Nombre del usuario
            
        Returns:
            Dict con la respuesta y metadatos
        """
        return self.query_processor.process_query(query, user_id, user_name)
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> list:
        """
        Recupera chunks relevantes usando el recuperador de contexto.
        
        Args:
            query: La consulta del usuario
            k: Número de chunks a recuperar
            
        Returns:
            Lista de chunks relevantes
        """
        return self.context_retriever.retrieve_relevant_chunks(query, k)
    
    def extract_keywords_from_query(self, query: str) -> list:
        """
        Extrae palabras clave usando el recuperador de contexto.
        
        Args:
            query: La consulta del usuario
            
        Returns:
            Lista de palabras clave
        """
        return self.context_retriever.extract_keywords_from_query(query)
    
    def enhance_context(self, query: str, context: str) -> str:
        """
        Mejora el contexto usando el recuperador de contexto.
        
        Args:
            query: La consulta original
            context: El contexto base
            
        Returns:
            Contexto mejorado
        """
        return self.context_retriever.enhance_context(query, context)
    
    def generate_response(self, query: str, context: str, sources: list = None) -> str:
        """
        Genera una respuesta usando el generador de respuestas.
        
        Args:
            query: La consulta del usuario
            context: Contexto de la consulta
            sources: Lista de fuentes consultadas
            
        Returns:
            Respuesta generada
        """
        return self.response_generator.generate_response(query, context, sources)
    
    def update_user_history(self, user_id: str, query: str, response: str):
        """
        Actualiza el historial del usuario.
        
        Args:
            user_id: ID del usuario
            query: La consulta realizada
            response: La respuesta generada
        """
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        self.user_history[user_id].append({
            'query': query,
            'response': response
        })
        
        # Mantener solo las últimas interacciones
        if len(self.user_history[user_id]) > self.max_history_length:
            self.user_history[user_id] = self.user_history[user_id][-self.max_history_length:]
        
        logger.info(f"Historial actualizado para usuario {user_id}")


# Para compatibilidad hacia atrás, mantener la clase original como alias
# (esto se puede remover en versiones futuras)
RAGSystemLegacy = RAGSystem