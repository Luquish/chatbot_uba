"""
Procesador de consultas para el sistema RAG.
Maneja la lógica principal de procesamiento de consultas.
"""
import uuid
from typing import Dict, Any
from config.settings import logger
from config.constants import SHEET_COURSE_KEYWORDS
from handlers.intent_handler import get_query_intent, handle_conversational_intent
from handlers.courses_handler import handle_sheet_course_query
from services.session_service import session_service


class QueryProcessor:
    """Procesador de consultas que maneja la lógica de ruteo e intención."""
    
    def __init__(self, router, sheets_service, cursos_spreadsheet_id, date_utils, 
                 normalized_intent_examples, update_history_callback):
        self.router = router
        self.sheets_service = sheets_service
        self.cursos_spreadsheet_id = cursos_spreadsheet_id
        self.date_utils = date_utils
        self.normalized_intent_examples = normalized_intent_examples
        self.update_history = update_history_callback
    
    def process_query(self, query: str, user_id: str = None, user_name: str = None) -> Dict[str, Any]:
        """
        Procesa una consulta y determina la respuesta apropiada.
        
        Args:
            query: La consulta del usuario
            user_id: ID único del usuario
            user_name: Nombre del usuario
            
        Returns:
            Dict con la respuesta y metadatos
        """
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        query_original = query
        query_lower = query.lower().strip()
        
        # Determinar intención usando keywords primero
        intent, confidence = get_query_intent(
            query_original, None, self.normalized_intent_examples
        )
        
        # Si no se detectó o confianza baja, usar patrones específicos
        intent, confidence = self._enhance_intent_detection(
            query_lower, intent, confidence
        )
        
        # Manejo de intenciones conversacionales
        conversational_response = handle_conversational_intent(
            intent, confidence, query_original, user_name
        )
        if conversational_response:
            return conversational_response
        
        # Construir contexto de sesión
        context = session_service.get_context_for_relative_query(user_id, query_original)
        
        # Verificar si es consulta de cursos
        if self._is_course_query(query_lower, context):
            return self._handle_course_query(query_original, query_lower, context, user_id)
        
        # Usar router para otras consultas
        try:
            result = self.router.route(query_original, context)
            if result and result.get("response"):
                return result
        except Exception as e:
            logger.error(f"Error en router: {str(e)}")
        
        # Fallback a respuesta por defecto
        return self._default_response(query_original)
    
    def _enhance_intent_detection(self, query_lower: str, intent: str, confidence: float) -> tuple:
        """Mejora la detección de intención con patrones específicos."""
        if intent == 'desconocido' or confidence < 0.6:
            # Patrones de cortesía
            if any(keyword in query_lower for keyword in 
                   ['como estas', 'cómo estás', 'como te va', 'todo bien']):
                intent, confidence = 'cortesia', 0.8
                logger.info(f"Intención detectada por patrones específicos: {intent} (confianza: {confidence:.2f})")
            
            # Patrones de identidad
            elif any(keyword in query_lower for keyword in 
                     ['como te llamas', 'cómo te llamás', 'tu nombre', 'quien eres']):
                intent, confidence = 'identidad', 0.8
                logger.info(f"Intención detectada por patrones específicos: {intent} (confianza: {confidence:.2f})")
            
            else:
                logger.info("Usando enfoque RAG estándar para la consulta")
        else:
            logger.info(f"Intención detectada por keywords: {intent} (confianza: {confidence:.2f})")
        
        return intent, confidence
    
    def _is_course_query(self, query_lower: str, context: Dict[str, Any]) -> bool:
        """Determina si la consulta es sobre cursos."""
        months_es = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 
                     'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
        
        mentions_month = any(month in query_lower for month in months_es)
        has_context_courses = context["query_type"] == "cursos"
        
        return (
            any(keyword in query_lower for keyword in SHEET_COURSE_KEYWORDS) or 
            (context["is_relative"] and context["query_type"] == "cursos") or
            (mentions_month and has_context_courses)
        )
    
    def _handle_course_query(self, query_original: str, query_lower: str, 
                           context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Maneja consultas específicas de cursos."""
        if not self.sheets_service:
            return self._default_response(query_original)
        
        # Consulta relativa con mes resuelto
        if context["is_relative"] and context["resolved_month"]:
            return self._handle_relative_course_query(query_original, context, user_id)
        
        # Consulta normal de cursos
        return self._handle_normal_course_query(query_original, query_lower, user_id)
    
    def _handle_relative_course_query(self, query_original: str, 
                                    context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Maneja consultas relativas de cursos."""
        logger.info(f"Procesando consulta relativa de cursos: {context['explanation']}")
        
        modified_query = f"cursos {context['resolved_month']}"
        sheet_course_response = handle_sheet_course_query(
            modified_query, self.sheets_service, self.cursos_spreadsheet_id, self.date_utils
        )
        
        session_service.update_session_context(
            user_id, query_original, "cursos", context["resolved_month"]
        )
        
        if sheet_course_response:
            response_with_context = f"📅 {context['explanation']}:\n\n{sheet_course_response}"
            self.update_history(user_id, query_original, response_with_context)
            return {
                "query": query_original, 
                "response": response_with_context,
                "query_type": "info_cursos_relativa",
                "relevant_chunks": [], 
                "sources": [f"Google Sheet Cursos - {context['resolved_month']}"]
            }
        
        return self._default_response(query_original)
    
    def _handle_normal_course_query(self, query_original: str, 
                                  query_lower: str, user_id: str) -> Dict[str, Any]:
        """Maneja consultas normales de cursos."""
        sheet_course_response = handle_sheet_course_query(
            query_original, self.sheets_service, self.cursos_spreadsheet_id, self.date_utils
        )
        
        if sheet_course_response:
            detected_month = self._detect_month_from_query(query_lower)
            
            if not detected_month:
                # Si no detectó mes específico, verificar referencias relativas
                current_date = self.date_utils.get_current_date()
                if any(term in query_lower for term in ['proxim', 'siguient', 'que viene']):
                    next_month = self.date_utils.get_next_month_name(current_date).upper()
                    detected_month = next_month
                elif 'este mes' in query_lower:
                    current_month = self.date_utils.get_current_month_name(current_date).upper()
                    detected_month = current_month
            
            # Actualizar contexto de sesión
            session_service.update_session_context(
                user_id, query_original, "cursos", detected_month
            )
            
            self.update_history(user_id, query_original, sheet_course_response)
            return {
                "query": query_original,
                "response": sheet_course_response,
                "query_type": "info_cursos",
                "relevant_chunks": [],
                "sources": ["Google Sheet Cursos"]
            }
        
        return self._default_response(query_original)
    
    def _detect_month_from_query(self, query_lower: str) -> str:
        """Detecta el mes mencionado en la consulta."""
        months_es = {
            'enero': 'ENERO', 'febrero': 'FEBRERO', 'marzo': 'MARZO', 'abril': 'ABRIL',
            'mayo': 'MAYO', 'junio': 'JUNIO', 'julio': 'JULIO', 'agosto': 'AGOSTO',
            'septiembre': 'SEPTIEMBRE', 'octubre': 'OCTUBRE', 'noviembre': 'NOVIEMBRE', 
            'diciembre': 'DICIEMBRE'
        }
        
        for month_name, month_upper in months_es.items():
            if month_name in query_lower:
                return month_upper
        
        return None
    
    def _default_response(self, query: str) -> Dict[str, Any]:
        """Genera respuesta por defecto cuando no se puede procesar la consulta."""
        return {
            "query": query,
            "response": "Lo siento, no pude procesar tu consulta. ¿Podrías reformularla?",
            "query_type": "error",
            "relevant_chunks": [],
            "sources": []
        }