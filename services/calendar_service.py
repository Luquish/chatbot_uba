"""
Servicio para interactuar con Google Calendar API.
Solo lectura de eventos públicos usando API Key.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytz

from config.calendar_config import CALENDAR_CONFIG, CALENDARS
from utils.date_utils import DateUtils

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
date_utils = DateUtils()

class CalendarService:
    def __init__(self):
        """Inicializa el servicio de Google Calendar."""
        if not CALENDAR_CONFIG['GOOGLE_API_KEY']:
            raise ValueError("Se requiere GOOGLE_API_KEY en la configuración")
        
        self.service = build('calendar', 'v3', 
                           developerKey=CALENDAR_CONFIG['GOOGLE_API_KEY'],
                           cache_discovery=False)
        self.timezone = pytz.timezone(CALENDAR_CONFIG['TIMEZONE'])
        logger.info("Servicio de Google Calendar inicializado correctamente")

    def get_calendar_id(self) -> Optional[str]:
        """Retorna el ID del calendario unificado de actividades CECIM."""
        info = CALENDARS.get('actividades_cecim')
        return info.get('id') if info else None

    def get_events_this_week(self) -> List[Dict]:
        """
        Obtiene los eventos de la semana actual.
        Args:
            calendar_type: Tipo de calendario específico (examenes, inscripciones, cursada, tramites)
        Returns:
            List[Dict]: Lista de eventos con formato simplificado
        """
        try:
            # Calcular inicio y fin de la semana
            now = datetime.now(self.timezone)
            start_of_week = now - timedelta(days=now.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            
            # Convertir a UTC para la API
            time_min = start_of_week.astimezone(pytz.UTC).isoformat()
            time_max = end_of_week.astimezone(pytz.UTC).isoformat()
            
            all_events = []
            
            calendar_id = self.get_calendar_id()
            if calendar_id:
                all_events = self._get_events_from_calendar(calendar_id, time_min, time_max)
            
            return self._format_events(all_events)
            
        except Exception as e:
            logger.error(f"Error al obtener eventos: {str(e)}")
            return []

    def _get_events_from_calendar(self, calendar_id: str, time_min: str, time_max: str) -> List[Dict]:
        """
        Obtiene eventos de un calendario específico.
        """
        try:
            logger.info(f"Buscando eventos en calendario: {calendar_id}")
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=CALENDAR_CONFIG['MAX_RESULTS'],
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            # Agregar el tipo de calendario a cada evento
            for event in events:
                event['calendar_type'] = 'actividades_cecim'
            
            return events
        except HttpError as error:
            logger.error(f"Error al obtener eventos del calendario {calendar_id}: {error}")
            return []

    def get_upcoming_events(self, max_results: int = 10) -> List[Dict]:
        """Obtiene próximos eventos del calendario unificado."""
        try:
            calendar_id = self.get_calendar_id()
            
            if not calendar_id:
                logger.error(f"No se encontró el calendario de actividades CECIM")
                return []
                
            logger.info(f"Buscando eventos próximos en el calendario de actividades CECIM")
            
            # Obtener eventos desde ahora
            now = datetime.now(self.timezone)
            
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            for event in events:
                event['calendar_type'] = 'actividades_cecim'
                
            formatted_events = self._format_events(events)
            logger.info(f"Se encontraron {len(formatted_events)} eventos en el calendario de actividades CECIM")
            
            return formatted_events
            
        except HttpError as error:
            logger.error(f"Error al buscar eventos en el calendario de actividades CECIM: {error}")
            return []
        except Exception as e:
            logger.error(f"Error inesperado al buscar eventos en el calendario de actividades CECIM: {e}")
            return []

    def get_events_by_date_range(self, start_date: datetime, end_date: datetime, max_results: int = None) -> List[Dict]:
        """
        Busca eventos en un rango de fechas específico.
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            max_results: Número máximo de resultados (opcional)
        """
        logger.info(f"Buscando eventos entre {start_date} y {end_date}")
        try:
            # Usar helper interno para respetar calendar_id y reusar formato
            calendar_id = self.get_calendar_id()
            if not calendar_id:
                logger.error("No se encontró calendar_id para actividades CECIM")
                return []
            events = self._get_events_from_calendar(
                calendar_id,
                date_utils.format_date_for_api(start_date),
                date_utils.format_date_for_api(end_date)
            )
            logger.info(f"Se encontraron {len(events)} eventos en el rango de fechas")
            return self._format_events(events)
            
        except HttpError as error:
            logger.error(f"Error al buscar eventos por rango de fechas: {error}")
            return []

    # get_upcoming_events redefinido arriba

    def get_events_from_query(self, query: str):
        """
        Obtiene eventos basados en una consulta en lenguaje natural.
        Args:
            query: Consulta en lenguaje natural
        """
        start_date, end_date = date_utils.extract_dates_from_query(query)
        
        # Si se encontraron fechas específicas en la consulta
        if start_date and end_date:
            return self.get_events_by_date_range(start_date, end_date)
            
        # Si la consulta menciona un tipo específico de evento
        tipos_evento = ['inscripción', 'inscripciones', 'examen', 'exámenes', 'parcial', 'parciales', 'final', 'finales']
        for tipo in tipos_evento:
            if tipo in query.lower():
                return self.get_events_by_type(tipo)
        
        # Por defecto, retorna los próximos eventos
        return self.get_upcoming_events()

    def _format_events(self, events: List[Dict]) -> List[Dict]:
        """
        Formatea una lista de eventos al formato simplificado.
        """
        if not events:
            return []
            
        # Mapeo de nombres de días en inglés a español
        days_map = {
            'Monday': 'Lunes',
            'Tuesday': 'Martes',
            'Wednesday': 'Miércoles',
            'Thursday': 'Jueves',
            'Friday': 'Viernes',
            'Saturday': 'Sábado',
            'Sunday': 'Domingo'
        }
            
        formatted_events = []
        for event in events:
            try:
                # Obtener fechas y convertir a zona horaria local
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Si es un evento de todo el día (solo fecha)
                if 'T' not in start:
                    start_dt = datetime.strptime(start, '%Y-%m-%d')
                    end_dt = datetime.strptime(end, '%Y-%m-%d')
                    
                    # Transformar a formato español
                    start_day = days_map.get(start_dt.strftime('%A'), start_dt.strftime('%A'))
                    end_day = days_map.get(end_dt.strftime('%A'), end_dt.strftime('%A'))
                    
                    start_formatted = f"{start_day} {start_dt.day} de {start_dt.strftime('%B')}"
                    end_formatted = f"{end_day} {end_dt.day} de {end_dt.strftime('%B')}"
                else:
                    # Convertir a zona horaria local
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    start_dt = start_dt.astimezone(self.timezone)
                    end_dt = end_dt.astimezone(self.timezone)
                    
                    # Transformar a formato español más amigable
                    start_day = days_map.get(start_dt.strftime('%A'), start_dt.strftime('%A'))
                    end_day = days_map.get(end_dt.strftime('%A'), end_dt.strftime('%A'))
                    
                    # Si es el mismo día
                    if start_dt.date() == end_dt.date():
                        start_formatted = f"{start_day} {start_dt.day} de {start_dt.hour}:{start_dt.minute:02d} hs a {end_dt.hour}:{end_dt.minute:02d} hs"
                    else:
                        start_formatted = f"{start_day} {start_dt.day} de {start_dt.hour}:{start_dt.minute:02d} hs"
                        end_formatted = f"{end_day} {end_dt.day} de {end_dt.hour}:{end_dt.minute:02d} hs"
                
                formatted_events.append({
                    'summary': event.get('summary', 'Sin título'),
                    'start': start_formatted,
                    'end': end_formatted if 'end_formatted' in locals() else None,  # Solo incluir end si es necesario
                    'description': event.get('description', ''),
                    'is_all_day': 'T' not in start,
                    'calendar_type': event.get('calendar_type', 'general'),
                    'same_day': start_dt.date() == end_dt.date() if 'T' in start else False
                })
            except Exception as e:
                logger.error(f"Error al formatear evento: {str(e)}")
                continue
        
        return formatted_events

# Función helper para uso directo
def get_this_weeks_events() -> List[Dict]:
    """
    Función helper para obtener eventos de la semana actual.
    
    Returns:
        List[Dict]: Lista de eventos formateados
    """
    try:
        calendar = CalendarService()
        return calendar.get_events_this_week()
    except Exception as e:
        logger.error(f"Error al obtener eventos de la semana: {str(e)}")
        return []

if __name__ == "__main__":
    # Ejemplo de uso
    events = get_this_weeks_events()
    for event in events:
        print(f"Evento: {event['summary']}")
        print(f"{'Fecha' if event['is_all_day'] else 'Inicio'}: {event['start']}")
        print(f"{'Hasta' if event['is_all_day'] else 'Fin'}: {event['end']}")
        if event['description']:
            print(f"Descripción: {event['description']}")
        print("---") 