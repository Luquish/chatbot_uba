"""
Manejador para consultas sobre el calendario académico.
"""
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

from config.constants import information_emojis
from services.calendar_service import CalendarService

logger = logging.getLogger(__name__)


def get_calendar_events(
    calendar_service: CalendarService,
    calendar_intent: str = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> str:
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
        # Estrategia:
        # - Si se pasa un rango explícito, usarlo
        # - Si no hay intención, usar "esta semana"
        # - Si hay intención sin rango, usar próximos eventos
        if start_date and end_date:
            events = calendar_service.get_events_by_date_range(start_date, end_date)
        else:
            events = calendar_service.get_events_this_week() if not calendar_intent else calendar_service.get_upcoming_events()
        
        if not events:
            return "No encontré eventos programados para este período."
            
        # Formatear respuesta
        response_parts = ["📅 Eventos encontrados:"]
        
        for event in events:
            summary = event['summary']
            start = event['start']
            is_same_day = event.get('same_day', False)
            event_type = event.get('calendar_type', 'actividades_cecim')
            event_emoji = '📌'
            
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