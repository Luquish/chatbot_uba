"""
Manejador para consultas sobre el calendario académico.
"""
import logging
import random
from typing import List, Dict, Any, Optional

from config.constants import CALENDAR_INTENT_MAPPING, CALENDAR_MESSAGES
from config.constants import information_emojis
from services.calendar_service import CalendarService

logger = logging.getLogger(__name__)


def get_calendar_events(calendar_service: CalendarService, calendar_intent: str = None) -> str:
    """
    Obtiene y formatea los eventos del calendario según la intención.
    
    Args:
        calendar_service (CalendarService): Servicio de calendario
        calendar_intent (str): Tipo de eventos a buscar (examenes, inscripciones, etc.)
            
    Returns:
        str: Mensaje formateado con los eventos encontrados
    """
    if not calendar_service:
        return "Lo siento, el servicio de calendario no está disponible en este momento."
            
    try:
        events = []
        
        # Si hay una intención específica, usar el método correspondiente
        if calendar_intent and calendar_intent in CALENDAR_INTENT_MAPPING:
            intent_config = CALENDAR_INTENT_MAPPING[calendar_intent]
            tool = intent_config['tool']
            
            if tool == 'get_events_by_type':
                calendar_type = intent_config['params']['calendar_type']
                events = calendar_service.get_events_by_type(calendar_type)
            elif tool == 'get_events_this_week':
                events = calendar_service.get_events_this_week()
            elif tool == 'get_events_by_date_range':
                # Implementar lógica para rango de fechas si es necesario
                events = calendar_service.get_events_by_date_range()
            elif tool == 'get_upcoming_events':
                events = calendar_service.get_upcoming_events()
        else:
            # Si no hay intención específica, mostrar eventos de la semana
            events = calendar_service.get_events_this_week()
        
        if not events:
            if calendar_intent and calendar_intent in CALENDAR_INTENT_MAPPING:
                return CALENDAR_INTENT_MAPPING[calendar_intent]['no_events_message']
            return "No encontré eventos programados para este período."
            
        # Formatear respuesta
        response_parts = ["📅 Eventos encontrados:"]
        
        for event in events:
            summary = event['summary']
            start = event['start']
            is_same_day = event.get('same_day', False)
            event_type = event.get('calendar_type', 'general')
            
            # Emoji según tipo de evento
            event_emoji = {
                'examenes': '📝',
                'inscripciones': '✍️',
                'cursada': '📚',
                'tramites': '📋'
            }.get(event_type, '📌')
            
            event_str = f"\n{event_emoji} {summary}"
            
            # Usar formato simplificado para todos los eventos
            if is_same_day:
                event_str += f"\n  {start}"
            elif event.get('end'):
                # Formato para eventos que duran más de un día
                event_str += f"\n  {start} a {event['end']}"
            else:
                event_str += f"\n  {start}"
            
            if event.get('description'):
                event_str += f"\n  Detalles: {event['description']}"
            
            response_parts.append(event_str)
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error al obtener eventos del calendario: {str(e)}")
        return "Lo siento, hubo un error al consultar los eventos del calendario." 