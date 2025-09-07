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
        
        # Verificar si es consulta relativa
        if context.get("is_relative", False):
            return self._handle_relative_query(query_original, context, user_id)
        
        # Verificar si es consulta de cursos
        if self._is_course_query(query_lower, context):
            return self._handle_course_query(query_original, query_lower, context, user_id)
        
        # Usar router para otras consultas
        try:
            result = self.router.route(query_original, context)
            if result and result.get("response"):
                # Corregir el query_type semántico antes de actualizar sesión y devolver
                semantic_query_type = self._get_semantic_query_type(query_original)
                if semantic_query_type:
                    result["query_type"] = semantic_query_type
                
                # Actualizar sesión con el contexto de la respuesta
                self._update_session_with_result(user_id, query_original, result)
                return result
        except Exception as e:
            logger.error(f"Error en router: {str(e)}")
        
        # Fallback a respuesta por defecto
        fallback_result = self._default_response(query_original)
        self._update_session_with_result(user_id, query_original, fallback_result)
        return fallback_result
    
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
    
    def _get_semantic_query_type(self, query: str) -> str:
        """Determina el tipo semántico de la consulta basado en su contenido."""
        query_lower = query.lower()
        
        # Detectar consultas de calendario
        calendar_keywords = [
            'actividades', 'eventos', 'semana', 'mes', 'calendario', 'agenda',
            'reuniones', 'charlas', 'talleres', 'jornadas', 'conferencias'
        ]
        
        # Detectar consultas de cursos
        course_keywords = [
            'cursos', 'materias', 'asignaturas', 'enero', 'febrero', 'marzo',
            'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre',
            'octubre', 'noviembre', 'diciembre'
        ]
        
        if any(keyword in query_lower for keyword in calendar_keywords):
            return "calendario_eventos_generales"
        elif any(keyword in query_lower for keyword in course_keywords):
            return "cursos"
        
        return None  # No hay tipo semántico específico
    
    def _update_session_with_result(self, user_id: str, query: str, result: Dict[str, Any]):
        """Actualiza la sesión con el contexto de la respuesta."""
        query_lower = query.lower()
        query_type = result.get("query_type", "")
        
        # Determinar el tipo de contexto basado en el contenido de la consulta
        # Esto es más preciso que usar solo el query_type del resultado
        
        # Detectar consultas de calendario
        calendar_keywords = [
            'actividades', 'eventos', 'semana', 'mes', 'calendario', 'agenda',
            'reuniones', 'charlas', 'talleres', 'jornadas', 'conferencias'
        ]
        
        # Detectar consultas de cursos
        course_keywords = [
            'cursos', 'materias', 'asignaturas', 'enero', 'febrero', 'marzo',
            'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre',
            'octubre', 'noviembre', 'diciembre'
        ]
        
        # Detectar si es consulta de calendario
        if any(keyword in query_lower for keyword in calendar_keywords):
            # Extraer referencia temporal de la consulta
            time_reference = "esta semana"  # Por defecto
            if "esta semana" in query_lower:
                time_reference = "esta semana"
            elif "próxima semana" in query_lower or "proxima semana" in query_lower:
                time_reference = "próxima semana"
            elif "este mes" in query_lower:
                time_reference = "este mes"
            elif "próximo mes" in query_lower or "proximo mes" in query_lower:
                time_reference = "próximo mes"
            
            session_service.update_session_context(
                user_id=user_id,
                query=query,
                query_type="calendario_eventos_generales",
                calendar_intent="eventos_generales",
                time_reference=time_reference
            )
            
        # Detectar si es consulta de cursos
        elif any(keyword in query_lower for keyword in course_keywords):
            # Extraer mes de la consulta
            month_requested = None
            months = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                     'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
            
            for month in months:
                if month in query_lower:
                    month_requested = month.upper()
                    break
            
            if not month_requested:
                month_requested = "AGOSTO"  # Por defecto
            
            session_service.update_session_context(
                user_id=user_id,
                query=query,
                query_type="cursos",
                month_requested=month_requested
            )
            
        else:
            # Consulta general - usar el query_type del resultado
            session_service.update_session_context(
                user_id=user_id,
                query=query,
                query_type=query_type
            )
    
    def _handle_relative_query(self, query_original: str, context: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Maneja consultas relativas detectadas por el SessionService."""
        query_type = context.get("query_type", "")
        resolved_month = context.get("resolved_month")
        resolved_time_reference = context.get("resolved_time_reference")
        
        # Determinar el tipo de consulta relativa
        if "calendario" in query_type:
            # Consulta relativa de calendario
            if resolved_time_reference:
                # Usar el CalendarTool con el contexto resuelto
                calendar_context = {
                    "last_query_type": query_type,
                    "last_calendar_intent": context.get("calendar_intent"),
                    "resolved_time_reference": resolved_time_reference
                }
                
                # Buscar el CalendarTool en el router
                for tool in self.router.tools:
                    if hasattr(tool, 'name') and tool.name == 'calendar':
                        result = tool.execute(query_original, {}, calendar_context)
                        if result and result.response:
                            response_result = {
                                "response": result.response,
                                "query_type": f"{query_type}_relativa",
                                "sources": result.sources or [],
                                "metadata": result.metadata or {},
                                "is_relative": True,
                                "resolved_context": resolved_time_reference
                            }
                            self._update_session_with_result(user_id, query_original, response_result)
                            return response_result
                        break
        
        elif "cursos" in query_type:
            # Consulta relativa de cursos - usar el método específico
            if resolved_month:
                return self._handle_relative_course_query(query_original, context, user_id)
        
        # Si no se pudo manejar específicamente, usar respuesta conversacional
        result = {
            "response": f"Entiendo que te refieres a {resolved_time_reference or resolved_month or 'algo específico'}, pero necesito más información para ayudarte mejor.",
            "query_type": f"{query_type}_relativa",
            "sources": [],
            "metadata": {"is_relative": True},
            "is_relative": True
        }
        
        # Actualizar sesión con el resultado
        self._update_session_with_result(user_id, query_original, result)
        return result
    
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
                "query_type": "cursos_relativa",
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
                "query_type": "cursos",
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