import os
import uuid
import random
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from unidecode import unidecode

from config.settings import logger, PRIMARY_MODEL, FALLBACK_MODEL, EMBEDDING_MODEL, MAX_OUTPUT_TOKENS, API_TIMEOUT, RAG_NUM_CHUNKS, SIMILARITY_THRESHOLD, GOOGLE_API_KEY, CURSOS_SPREADSHEET_ID
from config.constants import INTENT_EXAMPLES, GREETING_WORDS, information_emojis, greeting_emojis, warning_emojis, success_emojis, SHEET_COURSE_KEYWORDS, CALENDAR_INTENT_MAPPING, CALENDAR_MESSAGES
from models.openai_model import OpenAIModel, OpenAIEmbedding
from storage.vector_store import PostgreSQLVectorStore
from utils.date_utils import DateUtils
from handlers.intent_handler import (
    normalize_intent_examples,
    get_query_intent,
    handle_conversational_intent,
)
from handlers.courses_handler import handle_sheet_course_query
from handlers.calendar_handler import get_calendar_events
from handlers.faqs_handler import handle_faq_query
from services.calendar_service import CalendarService
from services.sheets_service import SheetsService
from services.session_service import session_service

load_dotenv()

class RAGSystem:
    def __init__(self):
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
        # Inicializar vector store con PostgreSQL/pgvector
        self.vector_store = PostgreSQLVectorStore(
            threshold=self.similarity_threshold
        )
        
        # Log de configuración
        logger.info("Vector store configurado con PostgreSQL/pgvector")
        logger.info(f"Umbral de similitud: {self.similarity_threshold}")
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
        
        # Delegar manejo de intenciones conversacionales al intent handler
        conversational = handle_conversational_intent(
            intent, confidence, query_original, user_name
        )
        if conversational:
            return conversational
            
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
        
        # Cursos (Google Sheets) - con soporte para contexto conversacional
        # Verificar PRIMERO si es una consulta relativa de cursos
        context = session_service.get_context_for_relative_query(user_id, query_original)
        
        # Si es una consulta relativa sobre cursos O contiene palabras clave de cursos O menciona un mes con contexto de cursos
        months_es = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
        mentions_month = any(month in query_lower for month in months_es)
        has_context_courses = context["query_type"] == "cursos"
        
        is_course_query = (
            any(keyword in query_lower for keyword in SHEET_COURSE_KEYWORDS) or 
            (context["is_relative"] and context["query_type"] == "cursos") or
            (mentions_month and has_context_courses)  # NUEVO: detectar meses con contexto de cursos
        )
        
        if self.sheets_service and is_course_query:
            if context["is_relative"] and context["resolved_month"]:
                # Es una consulta relativa con mes resuelto
                logger.info(f"Procesando consulta relativa de cursos: {context['explanation']}")
                
                # Modificar la consulta para incluir el mes resuelto
                modified_query = f"cursos {context['resolved_month']}"
                sheet_course_response = handle_sheet_course_query(
                    modified_query, self.sheets_service, self.CURSOS_SPREADSHEET_ID, self.date_utils
                )
                
                # Actualizar contexto con la nueva consulta
                session_service.update_session_context(
                    user_id, query_original, "cursos", context["resolved_month"]
                )
                
                if sheet_course_response:
                    # Personalizar la respuesta para indicar que se interpretó la consulta relativa
                    response_with_context = f"📅 {context['explanation']}:\n\n{sheet_course_response}"
                    self.update_user_history(user_id, query_original, response_with_context)
                    return {
                        "query": query_original, "response": response_with_context,
                        "query_type": "info_cursos_relativa",
                        "relevant_chunks": [], "sources": [f"Google Sheet Cursos - {context['resolved_month']}"]
                    }
            
            # Consulta normal de cursos
            sheet_course_response = handle_sheet_course_query(
                query_original, self.sheets_service, self.CURSOS_SPREADSHEET_ID, self.date_utils
            )
            if sheet_course_response:
                # Detectar qué mes se consultó para actualizar el contexto
                query_lower_for_month = query_original.lower()
                months_es = {
                    'enero': 'ENERO', 'febrero': 'FEBRERO', 'marzo': 'MARZO', 'abril': 'ABRIL',
                    'mayo': 'MAYO', 'junio': 'JUNIO', 'julio': 'JULIO', 'agosto': 'AGOSTO',
                    'septiembre': 'SEPTIEMBRE', 'octubre': 'OCTUBRE', 'noviembre': 'NOVIEMBRE', 'diciembre': 'DICIEMBRE'
                }
                
                detected_month = None
                for month_name, month_upper in months_es.items():
                    if month_name in query_lower_for_month:
                        detected_month = month_upper
                        break
                
                # Si no se detectó un mes específico, verificar referencias relativas
                if not detected_month:
                    if any(phrase in query_lower_for_month for phrase in ["mes que viene", "próximo mes", "proximo mes", "siguiente mes"]):
                        # Obtener el mes siguiente
                        current_month = self.date_utils.get_today().month
                        next_month = current_month + 1 if current_month < 12 else 1
                        detected_month = list(months_es.values())[next_month - 1]
                    else:
                        # Usar el mes actual como fallback
                        detected_month = self.date_utils.get_current_month_name()
                
                # Actualizar contexto
                session_service.update_session_context(
                    user_id, query_original, "cursos", detected_month
                )
                
                self.update_user_history(user_id, query_original, sheet_course_response)
                return {
                    "query": query_original, "response": sheet_course_response,
                    "query_type": "info_cursos",
                    "relevant_chunks": [], "sources": [f"Google Sheet Cursos"]
                }
        # Calendario - con soporte para contexto conversacional
        # Verificar si es una consulta relativa de calendario O contiene palabras clave de calendario
        is_calendar_query = False
        calendar_intent = None
        
        # Si es una consulta relativa sobre calendario
        if context["is_relative"] and context["query_type"].startswith("calendario") and context["resolved_time_reference"]:
            is_calendar_query = True
            calendar_intent = context["calendar_intent"]  # Usar el intent anterior
            
            logger.info(f"Procesando consulta relativa de calendario: {context['explanation']}")
            
            # Crear consulta modificada con la referencia temporal resuelta
            modified_query = f"eventos {context['resolved_time_reference']}"
            calendar_response = get_calendar_events(self.calendar_service, calendar_intent)
            
            # Actualizar contexto con la nueva consulta
            session_service.update_session_context(
                user_id, query_original, f"calendario_{calendar_intent}", 
                calendar_intent=calendar_intent, 
                time_reference=context["resolved_time_reference"]
            )
            
            if calendar_response:
                # Personalizar la respuesta para indicar que se interpretó la consulta relativa
                response_with_context = f"📅 {context['explanation']}:\n\n{calendar_response}"
                self.update_user_history(user_id, query_original, response_with_context)
                return {
                    "query": query_original, "response": response_with_context,
                    "query_type": f"calendario_{calendar_intent}_relativa",
                    "relevant_chunks": [], "sources": [f"Calendario Académico - {context['resolved_time_reference']}"]
                }
        
        # Consulta normal de calendario (detectar por palabras clave O mes con contexto de calendario)
        has_context_calendar = context["query_type"].startswith("calendario")
        
        if not is_calendar_query:
            # Detectar por palabras clave de calendario
            for intent, config in CALENDAR_INTENT_MAPPING.items():
                if any(keyword in query_lower for keyword in config['keywords']):
                    calendar_intent = intent
                    is_calendar_query = True
                    break
            
            # También detectar si menciona un mes con contexto de calendario
            if not is_calendar_query and mentions_month and has_context_calendar:
                is_calendar_query = True
                calendar_intent = context["calendar_intent"]  # Usar el intent del contexto anterior
                logger.info(f"Consulta de calendario detectada por mes con contexto: {calendar_intent}")
        
        if is_calendar_query and calendar_intent:
            calendar_response = get_calendar_events(self.calendar_service, calendar_intent)
            
            # Detectar referencia temporal para guardar en el contexto
            time_reference = None
            if "esta semana" in query_lower:
                time_reference = "esta semana"
            elif "este mes" in query_lower:
                time_reference = "este mes"
            elif "próximos" in query_lower or "proximos" in query_lower:
                time_reference = "próximos eventos"
            
            # Actualizar contexto
            session_service.update_session_context(
                user_id, query_original, f"calendario_{calendar_intent}",
                calendar_intent=calendar_intent,
                time_reference=time_reference
            )
            
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
            
        # Normalizar consulta
        query_lower = query.lower().strip()
        
        # Sistema de detección de "consultas críticas" basado en entidades clave
        # Define entidades críticas que deben ser priorizadas (más general que hardcodear solo denuncias)
        critical_entities = {
            'denuncia': {
                'keywords': ['denuncia', 'denunciar', 'denuncias'],
                'context_words': ['presentar', 'cómo', 'como', 'donde', 'dónde', 'procedimiento'],
                'article_patterns': ['art. 5', 'artículo 5', 'art. 5º', 'artículo 5º'],
                'priority': 0.95,  # Prioridad muy alta
                'secondary_priority': 0.80  # Prioridad para menciones secundarias
            },
            'regimen_disciplinario': {
                'keywords': ['régimen disciplinario', 'regimen disciplinario', 'disciplina', 'sanción', 'sancion'],
                'context_words': ['suspensión', 'suspension', 'aplazo', 'falta', 'sumario'],
                'article_patterns': [],
                'priority': 0.90,
                'secondary_priority': 0.75
            },
            'regularidad': {
                'keywords': ['regularidad', 'regular', 'condiciones'],
                'context_words': ['alumno', 'estudiante', 'requisito', 'mantener', 'perder'],
                'article_patterns': [],
                'priority': 0.90,
                'secondary_priority': 0.75
            }
            # Se pueden añadir más entidades críticas aquí sin modificar la lógica general
        }
        
        # Detectar si es una consulta sobre una entidad crítica
        detected_entities = []
        for entity_name, entity_data in critical_entities.items():
            # Verificar keywords principales
            if any(keyword in query_lower for keyword in entity_data['keywords']):
                # Si también contiene palabras de contexto, es aún más relevante
                if any(context in query_lower for context in entity_data['context_words']):
                    detected_entities.append((entity_name, entity_data, True))  # True = alta prioridad
                else:
                    detected_entities.append((entity_name, entity_data, False))  # False = prioridad estándar
        
        # Realizar búsqueda especializada para entidades críticas detectadas
        priority_chunks = []
        if detected_entities:
            logger.info(f"Detectadas entidades críticas: {[e[0] for e in detected_entities]}")
            
            # Buscar chunks relacionados con las entidades detectadas
            for entity_name, entity_data, is_high_priority in detected_entities:
                logger.info(f"Realizando búsqueda prioritaria para entidad: {entity_name}")
                
                # Buscar chunks que contengan información relevante sobre esta entidad
                # Nota: PostgreSQLVectorStore no tiene metadata_df, se usa búsqueda por embedding
                logger.info(f"Búsqueda por embedding para entidad: {entity_name}")
                # La búsqueda se realizará por embedding en lugar de metadata_df
        
        # Realizar búsqueda principal por embedding
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.vector_store.search(query_embedding, k=k)
        
        # Añadir chunks prioritarios si se detectaron entidades críticas
        if priority_chunks:
            # Añadir solo chunks que no estén ya en los resultados
            for chunk in priority_chunks:
                if not any(r.get('text') == chunk.get('text') for r in results):
                    results.insert(0, chunk)  # Insertar al inicio para máxima prioridad
            
            # Limitar los resultados si exceden k+3
            if len(results) > k+3:
                results = results[:k+3]
                
            logger.info(f"Se agregaron {len(priority_chunks)} chunks prioritarios")
        
        # Extracción de palabras clave de la consulta
        important_keywords = self.extract_keywords_from_query(query_lower)
        logger.info(f"Palabras clave extraídas: {important_keywords}")
        
        # Si no hay suficientes resultados o la similitud es baja, intentar complementar con palabras clave
        if len(results) < 3 or (results and results[-1]['similarity'] < self.similarity_threshold + 0.05):
            logger.info(f"Resultados iniciales insuficientes, intentando complementar con palabras clave: {important_keywords}")
            
            # Intentar búsquedas adicionales solo con las palabras clave más importantes
            for keyword in important_keywords[:3]:  # Usar solo las 3 palabras clave más importantes
                # Buscar chunks que contengan esta palabra clave usando búsqueda por embedding
                logger.info(f"Búsqueda adicional por palabra clave: {keyword}")
                # Nota: PostgreSQLVectorStore no tiene metadata_df, se usa búsqueda por embedding
                # La búsqueda se realizará por embedding en lugar de metadata_df
        
        # Log de resultados encontrados
        logger.info(f"Número total de chunks recuperados: {len(results)}")
        for i, r in enumerate(results):
            filename = r.get('filename', 'unknown')
            similarity = r.get('similarity', 0)
            text_preview = r.get('text', '')[:100] + '...' if len(r.get('text', '')) > 100 else r.get('text', '')
            logger.info(f"Chunk {i+1}: {filename} (similitud: {similarity:.2f}) - {text_preview}")
        
        # Ordenar por similitud para mantener los mejores resultados primero
        results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Limitar al número máximo de resultados
        results = results[:k+3]  # Permitir hasta 3 resultados adicionales
        
        return results
        
    def extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extrae palabras clave relevantes de la consulta.
        
        Args:
            query (str): Consulta normalizada
            
        Returns:
            List[str]: Lista de palabras clave ordenadas por relevancia
        """
        # Limpieza adicional para eliminar signos de puntuación
        query = re.sub(r'[^\w\s]', ' ', query).strip()
        
        # Palabras comunes en español a ignorar
        stopwords = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'es', 'son', 'y', 'o', 
                   'a', 'ante', 'bajo', 'con', 'contra', 'de', 'desde', 'en', 'entre', 'hacia', 
                   'hasta', 'para', 'por', 'segun', 'sin', 'sobre', 'tras', 'como', 'que', 'quien',
                   'donde', 'cuando', 'cuanto', 'cual', 'cuales', 'cuantos', 'cuantas', 'me', 'mi',
                   'tu', 'te', 'se', 'nos', 'le', 'lo', 'su', 'sus', 'esto', 'eso', 'aquello', 'este',
                   'ese', 'aquel', 'esta', 'esa', 'aquella', 'estos', 'esos', 'aquellos', 'estas', 
                   'esas', 'aquellas']
        
        # Dividir en palabras
        words = query.split()
        
        # Filtrar stopwords y palabras cortas
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Definir palabras clave del dominio por categorías
        domain_keywords = {
            # Términos relacionados con normativas y reglamentos
            'normativas': {
                'weight': 10,
                'terms': ['regimen', 'disciplinario', 'reglamento', 'normativa', 'articulo', 
                          'resolución', 'regulación', 'estatuto', 'disposición']
            },
            # Términos relacionados con denuncias y trámites
            'denuncias': {
                'weight': 15,
                'terms': ['denuncia', 'denuncias', 'denunciar', 'presentar', 'tramitar', 
                          'procedimiento', 'formular']
            },
            # Términos relacionados con la condición de estudiante
            'condicion_estudiante': {
                'weight': 10,
                'terms': ['regularidad', 'regular', 'condicion', 'condiciones', 'alumno', 
                          'estudiante', 'readmision', 'reincorporacion']
            },
            # Términos relacionados con cursadas y materias
            'cursada': {
                'weight': 9,
                'terms': ['materias', 'asignaturas', 'aprobadas', 'aprobar', 'inscripcion', 
                          'inscribirse', 'recursar', 'aplazo', 'aplazos']
            },
            # Términos relacionados con sanciones
            'sanciones': {
                'weight': 10,
                'terms': ['sancion', 'sanciones', 'suspension', 'sumario', 'falta', 'faltas']
            },
            # Términos de navegación y búsqueda
            'navegacion': {
                'weight': 8,
                'terms': ['donde', 'dónde', 'como', 'cómo', 'manera', 'forma', 'cuando', 
                          'cuándo', 'quien', 'quién', 'requisitos']
            }
        }

        # Detectar patrones de preguntas específicas sobre procedimientos
        procedure_patterns = [
            # Patrones para "cómo hacer algo"
            r'(como|cómo).*?(hacer|presentar|realizar|tramitar|inscribir)',
            # Patrones para "dónde hacer algo"
            r'(donde|dónde).*?(presentar|ir|entregar|solicitar|tramitar)',
            # Patrones para "cuál es el procedimiento"
            r'(cual|cuál).*?(procedimiento|forma|trámite|proceso)',
            # Patrones para "qué necesito para"
            r'(que|qué).*?(necesito|requiero|debo).*?(para|hacer)',
            # Patrones genéricos de solicitud
            r'(quiero|deseo|necesito).*?(hacer|presentar|solicitar|tramitar)'
        ]
        
        # Detectar si es una consulta sobre algún procedimiento
        is_procedure_query = False
        for pattern in procedure_patterns:
            if re.search(pattern, query.lower()):
                is_procedure_query = True
                logger.info(f"Detectado patrón de procedimiento: {pattern}")
                break
        
        # Asignar pesos a las palabras clave según su categoría
        weighted_keywords = []
        for word in keywords:
            weight_assigned = False
            
            # Buscar en cada categoría
            for category, data in domain_keywords.items():
                if word in data['terms']:
                    # Si es pregunta de procedimiento, dar más peso a palabras de procedimientos
                    extra_weight = 5 if is_procedure_query and category in ['denuncias', 'navegacion'] else 0
                    weighted_keywords.append((word, data['weight'] + extra_weight))
                    weight_assigned = True
                    break
            
            # Si no se encontró en ninguna categoría, asignar peso predeterminado
            if not weight_assigned:
                weighted_keywords.append((word, 5))
        
        # Ordenar por importancia
        weighted_keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Asegurar que las palabras más importantes estén bien representadas
        # Si no hay palabras clave del dominio pero se detecta un patrón de procedimiento,
        # añadir "procedimiento" como palabra clave
        if is_procedure_query and not any(w[1] > 5 for w in weighted_keywords):
            weighted_keywords.insert(0, ('procedimiento', 10))
        
        # Devolver solo las palabras
        return [word for word, _ in weighted_keywords]

    def enhance_context(self, query: str, context: str) -> str:
        """
        Mejora el contexto para destacar la información más relevante para la consulta.
        
        Args:
            query (str): Consulta del usuario
            context (str): Contexto recuperado del sistema RAG
            
        Returns:
            str: Contexto mejorado con información destacada
        """
        # Sistema de detección de "tipos de consulta" para personalizar el contexto
        query_types = {
            'procedimiento': {
                'patterns': [
                    r'(como|cómo).*?(hacer|presentar|realizar|obtener)',
                    r'(donde|dónde).*?(presentar|hacer|obtener|solicitar)',
                    r'(cual|cuál).*?(procedimiento|forma|manera|paso)',
                ],
                'keywords': ['procedimiento', 'paso', 'requisito', 'tramitar', 'presentar', 'realizar'],
                'highlight': 'PROCEDIMIENTO:'
            },
            'normativa': {
                'patterns': [
                    r'(que|qué).*?(dice|establece|indica).*?(reglamento|normativa|régimen)',
                    r'(artículo|art).*?(\d+)',
                    r'(según|segun).*?(reglamento|normativa|régimen)',
                ],
                'keywords': ['reglamento', 'normativa', 'régimen', 'disciplinario', 'artículo', 'resolución'],
                'highlight': 'NORMATIVA APLICABLE:'
            },
            'denuncia': {
                'patterns': [
                    r'(denuncia|denunciar|denuncias)',
                    r'(presentar|hacer).*?(denuncia|denuncias)',
                ],
                'keywords': ['denuncia', 'denunciar', 'denuncias'],
                'highlight': 'INFORMACIÓN SOBRE DENUNCIAS:'
            },
            'regularidad': {
                'patterns': [
                    r'(regularidad|regular|condición|condicion).*?(alumno|estudiante)',
                    r'(perder|mantener|recuperar).*?(regularidad|condición|condicion)',
                ],
                'keywords': ['regularidad', 'regular', 'condición', 'estudiante', 'alumno'],
                'highlight': 'CONDICIONES DE REGULARIDAD:'
            }
        }
        
        # Detectar el tipo de consulta
        detected_types = []
        for query_type, data in query_types.items():
            # Verificar patrones
            for pattern in data['patterns']:
                if re.search(pattern, query.lower()):
                    detected_types.append(query_type)
                    logger.info(f"Detectado tipo de consulta: {query_type} (patrón: {pattern})")
                    break
            
            # Si no se detectó por patrón, verificar keywords
            if query_type not in detected_types:
                if any(keyword in query.lower() for keyword in data['keywords']):
                    detected_types.append(query_type)
                    logger.info(f"Detectado tipo de consulta: {query_type} (keywords)")
        
        # Extraer palabras clave de la consulta
        keywords = self.extract_keywords_from_query(query.lower())
        
        # Dividir el contexto en párrafos
        paragraphs = context.split('\n\n')
        
        # Identificar si la consulta busca información específica sobre artículos
        seeking_articles = any(word in query.lower() for word in ['articulo', 'artículo', 'art', 'apartado', 'inciso', 'punto'])
        
        # Patrones para detectar referencias a artículos
        article_patterns = [
            r'Art\. \d+[º°]?\.', 
            r'Artículo \d+[º°]?\.', 
            r'Artículo \d+[º°]?:', 
            r'Art\. \d+[º°]?:',
            r'inc\. [a-z]\)'
        ]
        
        # Para cada párrafo, verificar si contiene información relevante
        enhanced_paragraphs = []
        for paragraph in paragraphs:
            relevance_score = 0
            
            # Verificar si el párrafo contiene palabras clave de la consulta
            for keyword in keywords:
                if keyword in paragraph.lower():
                    relevance_score += 1
            
            # Verificar si el párrafo menciona algún artículo
            has_article = False
            specific_article = None
            for pattern in article_patterns:
                article_match = re.search(pattern, paragraph)
                if article_match:
                    has_article = True
                    relevance_score += 2
                    specific_article = article_match.group(0)
            
            # Si la consulta busca artículos específicamente, dar más peso a párrafos con artículos
            if seeking_articles and has_article:
                relevance_score += 3
            
            # Dar más peso a párrafos relevantes para los tipos de consulta detectados
            for query_type in detected_types:
                if any(keyword in paragraph.lower() for keyword in query_types[query_type]['keywords']):
                    relevance_score += 3
                    
                    # Destacar especialmente este párrafo con un prefijo
                    if relevance_score >= 5:
                        enhanced_paragraphs.append(f"{query_types[query_type]['highlight']}\n{paragraph}")
                        break
            else:  # Este else pertenece al for, se ejecuta si no hubo break
                # Destacar párrafos muy relevantes
                if relevance_score >= 4:
                    enhanced_paragraphs.append(f"INFORMACIÓN RELEVANTE:\n{paragraph}")
                elif relevance_score > 0:
                    enhanced_paragraphs.append(paragraph)
                elif has_article:  # Incluir todos los artículos, incluso si no parecen directamente relevantes
                    enhanced_paragraphs.append(paragraph)
        
        # Si no hay párrafos con relevancia, usar el contexto original
        if not enhanced_paragraphs:
            return context
            
        # Buscar artículos específicos importantes si no fueron incluidos
        # Esto es más general que solo buscar el Artículo 5 para denuncias
        for query_type in detected_types:
            if query_type in ['normativa', 'denuncia', 'regularidad'] and len(enhanced_paragraphs) < 5:
                for paragraph in paragraphs:
                    if paragraph not in enhanced_paragraphs:
                        # Buscar artículos relevantes según el tipo de consulta
                        relevant_keywords = query_types[query_type]['keywords']
                        if has_article and any(keyword in paragraph.lower() for keyword in relevant_keywords):
                            enhanced_paragraphs.append(paragraph)
                            break
            
        # Unir párrafos destacados
        enhanced_context = '\n\n'.join(enhanced_paragraphs)
        
        # Añadir un mensaje introductorio relevante según el tipo de consulta
        intro_messages = {
            'procedimiento': "A continuación se detalla el procedimiento según la normativa oficial:",
            'normativa': "A continuación se citan las disposiciones relevantes de la normativa:",
            'denuncia': "La siguiente información explica el procedimiento oficial para presentar denuncias:",
            'regularidad': "A continuación se detallan las condiciones de regularidad según la normativa vigente:"
        }
        
        if detected_types and any(intro_messages.get(t) for t in detected_types):
            # Usar el primer tipo detectado que tenga mensaje introductorio
            for t in detected_types:
                if t in intro_messages:
                    enhanced_context = f"{intro_messages[t]}\n\n{enhanced_context}"
                    break
        
        return enhanced_context

    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        emoji = random.choice(information_emojis)
        
        # Preparar la lista de fuentes
        sources_text = ""
        if sources and len(sources) > 0:
            unique_sources = list(set(sources))  # Eliminar duplicados
            sources_text = f"\nFUENTES CONSULTADAS:\n{', '.join(unique_sources)}"
        
        # Analizar el contexto y la consulta para dar mejor estructura a la respuesta
        context_improved = self.enhance_context(query, context)
        
        # Sistema de clasificación de consultas para personalizar instrucciones al modelo
        query_classifiers = {
            'denuncias': {
                'keywords': ['denuncia', 'denunciar', 'denuncias'],
                'patterns': [
                    r'(como|cómo|de que forma|donde|dónde).*?(presentar?|hacer|poner|realizar|tramitar).*?(denuncia)',
                    r'(presentar?|hacer|poner|realizar|tramitar).*?(denuncia)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE DENUNCIAS:
- Menciona EXPLÍCITAMENTE que las denuncias se presentan POR ESCRITO
- Indica claramente que debe incluir una relación circunstanciada de hechos y personas
- Menciona la alternativa de presentación verbal en casos de urgencia
- Indica que la denuncia verbal debe ratificarse por escrito dentro de las 48 horas
- Si hay artículos relevantes en la información, cítalos textualmente
- Asegúrate de mencionar que la Universidad puede iniciar sumarios de oficio
"""
            },
            'regimen_disciplinario': {
                'keywords': ['régimen disciplinario', 'regimen disciplinario', 'disciplina', 'sanción', 'sancion'],
                'patterns': [
                    r'(sancion|sanción|castigo|pena|amonestacion|expulsión|suspension).*?(estudiante|alumno)',
                    r'(que|qué).*?(sancion|sanción|castigo|pena).*?(corresponde|aplica)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE RÉGIMEN DISCIPLINARIO:
- Cita con precisión los artículos relevantes del Régimen Disciplinario
- Incluye los tipos de sanciones que pueden aplicarse y su graduación
- Menciona qué autoridades pueden aplicar cada tipo de sanción
- Si se mencionan plazos o procedimientos específicos, destácalos claramente
- Explica claramente qué derechos tiene el estudiante en un proceso disciplinario
"""
            },
            'regularidad': {
                'keywords': ['regularidad', 'regular', 'condiciones'],
                'patterns': [
                    r'(como|cómo).*?(mantener|conseguir|obtener|perder).*?(regularidad|condición|condicion)',
                    r'(requisito|requisitos).*?(alumno regular|regularidad|condición)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE REGULARIDAD:
- Destaca claramente el número mínimo de materias a aprobar en cada período
- Menciona el porcentaje máximo de aplazos permitidos
- Explica los plazos establecidos para completar la carrera
- Si hay excepciones o situaciones especiales, menciónalas
- Cita los artículos específicos sobre regularidad que sean relevantes
"""
            }
        }
        
        # Detectar tipo de consulta
        specific_instructions = ""
        for category, data in query_classifiers.items():
            # Verificar por keywords
            if any(keyword in query.lower() for keyword in data['keywords']):
                specific_instructions = data['instructions']
                break
                
            # Verificar por patrones más complejos
            if not specific_instructions:
                for pattern in data['patterns']:
                    if re.search(pattern, query.lower()):
                        specific_instructions = data['instructions']
                        break
                        
        prompt = f"""
Sos DrCecim, un asistente virtual especializado de la Facultad de Medicina UBA. Tu tarea es proporcionar respuestas sobre administración y trámites de la facultad y deben ser breves, precisas y útiles.

INFORMACIÓN RELEVANTE:
{context_improved}
{sources_text}

CONSULTA ACTUAL: {query}
{specific_instructions}

RESPONDE SIGUIENDO ESTAS REGLAS:
1. Sé muy conciso y directo
2. Usa la información de los documentos oficiales proporcionados
3. Si hay artículos, resoluciones o reglamentos específicos, cita exactamente el número y fuente 
4. No omitas información importante de los documentos relevantes
5. Si hay documentos específicos, cita naturalmente su origen ("Según el reglamento...")
6. NO uses formato Markdown ya que esto no se procesa correctamente en mensajería
7. Para enfatizar texto, usa MAYÚSCULAS o comillas
8. Usa viñetas con guiones (-) cuando sea útil para organizar información
9. Si la información está incompleta, sugiere contactar a @cecim.nemed por instagram
10. No inventes o asumas información que no esté en los documentos
11. No hagas preguntas adicionales en tu respuesta
"""
        try:
            response = self.model.generate(prompt)
        except Exception as e:
            logger.error(f"Error al generar respuesta con el modelo primario: {str(e)}")
            try:
                # Intentar con el modelo de respaldo
                logger.info(f"Intentando con el modelo de respaldo: {self.fallback_model_name}")
                fallback_model = OpenAIModel(
                    model_name=self.fallback_model_name,
                    api_key=self.openai_api_key,
                    timeout=self.api_timeout,
                    max_output_tokens=self.max_output_tokens
                )
                response = fallback_model.generate(prompt)
            except Exception as e2:
                logger.error(f"Error también con el modelo de respaldo: {str(e2)}")
                return f"{emoji} Lo siento, hubo un error al generar la respuesta. Por favor, intenta de nuevo o contacta a @cecim.nemed por instagram."
                
        # Limpiar formatos que no se procesan bien en mensajería
        response = re.sub(r'\*\*(.+?)\*\*', r'\1', response)
        response = re.sub(r'\*(.+?)\*', r'\1', response)
        response = re.sub(r'\_\_(.+?)\_\_', r'\1', response)
        response = re.sub(r'\_(.+?)\_', r'\1', response)
        
        # Agregar emoji si no tiene uno al inicio
        if not re.match(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]', response.strip()[0:1]):
            response = f"{emoji} {response}"
        
        # Comprobar la calidad de la respuesta para tipos específicos de consultas
        for category, data in query_classifiers.items():
            if any(keyword in query.lower() for keyword in data['keywords']) or any(re.search(pattern, query.lower()) for pattern in data['patterns']):
                # Lista de comprobaciones específicas para categorías principales
                if category == 'denuncias' and "por escrito" not in response.lower():
                    logger.warning("La respuesta sobre denuncias no incluye información sobre presentación por escrito")
                    response += "\n\nRECUERDA: Las denuncias DEBEN presentarse POR ESCRITO con todos los detalles relevantes."
                    
                elif category == 'regimen_disciplinario' and not any(keyword in response.lower() for keyword in ['apercibimiento', 'suspensión', 'sanción']):
                    logger.warning("La respuesta sobre régimen disciplinario no menciona tipos de sanciones")
                    
                elif category == 'regularidad' and not any(term in response.lower() for term in ['materias', 'aprobar', 'porcentaje', 'plazo']):
                    logger.warning("La respuesta sobre regularidad no incluye información sobre requisitos clave")
            
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