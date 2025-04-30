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
import sys

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.calendar_config import CALENDAR_CONFIG, CALENDARS
from scripts.date_utils import DateUtils

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
date_utils = DateUtils()

class CalendarService:
    def __init__(self):
        """Inicializa el servicio de Google Calendar."""
        if not CALENDAR_CONFIG['GOOGLE_CALENDAR_API_KEY']:
            raise ValueError("Se requiere GOOGLE_CALENDAR_API_KEY en la configuración")
        
        self.service = build('calendar', 'v3', 
                           developerKey=CALENDAR_CONFIG['GOOGLE_CALENDAR_API_KEY'],
                           cache_discovery=False)
        self.timezone = pytz.timezone(CALENDAR_CONFIG['TIMEZONE'])
        logger.info("Servicio de Google Calendar inicializado correctamente")

    def get_calendar_id_by_type(self, calendar_type: str) -> Optional[str]:
        """
        Obtiene el ID del calendario según su tipo.
        Args:
            calendar_type: Tipo de calendario (examenes, inscripciones, cursada, tramites)
        Returns:
            str: ID del calendario o None si no se encuentra
        """
        calendar_info = CALENDARS.get(calendar_type.lower())
        if calendar_info and calendar_info['id']:
            logger.info(f"Calendario encontrado para tipo {calendar_type}")
            return calendar_info['id']
        logger.error(f"No se encontró el calendario para tipo: {calendar_type}")
        return None

    def get_events_this_week(self, calendar_type: str = None) -> List[Dict]:
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
            
            # Si se especifica un tipo, buscar solo en ese calendario
            if calendar_type:
                calendar_id = self.get_calendar_id_by_type(calendar_type)
                if calendar_id:
                    events = self._get_events_from_calendar(calendar_id, time_min, time_max)
                    if events:
                        for event in events:
                            event['calendar_type'] = calendar_type
                        all_events.extend(events)
            else:
                # Buscar en todos los calendarios
                for cal_type, cal_info in CALENDARS.items():
                    if cal_info['id']:
                        events = self._get_events_from_calendar(cal_info['id'], time_min, time_max)
                        if events:
                            for event in events:
                                event['calendar_type'] = cal_type
                            all_events.extend(events)
            
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
            calendar_type = next((cal_type for cal_type, info in CALENDARS.items() 
                               if info['id'] == calendar_id), None)
            if calendar_type:
                for event in events:
                    event['calendar_type'] = calendar_type
            
            return events
        except HttpError as error:
            logger.error(f"Error al obtener eventos del calendario {calendar_id}: {error}")
            return []

    def get_events_by_type(self, calendar_type: str, max_results: int = 10) -> List[Dict]:
        """
        Busca eventos en un calendario específico.
        Args:
            calendar_type: Tipo de calendario (examenes, inscripciones, cursada, tramites)
            max_results: Número máximo de resultados a retornar
        """
        try:
            # Normalizar el tipo de calendario
            calendar_type = calendar_type.lower().strip()
            calendar_id = self.get_calendar_id_by_type(calendar_type)
            
            if not calendar_id:
                logger.error(f"No se encontró el calendario: {calendar_type}")
                return []
                
            logger.info(f"Buscando eventos en el calendario {calendar_type}")
            
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
            
            # Agregar el tipo de calendario a cada evento
            for event in events:
                event['calendar_type'] = calendar_type
                
            formatted_events = self._format_events(events)
            logger.info(f"Se encontraron {len(formatted_events)} eventos en el calendario {calendar_type}")
            
            return formatted_events
            
        except HttpError as error:
            logger.error(f"Error al buscar eventos en el calendario {calendar_type}: {error}")
            return []
        except Exception as e:
            logger.error(f"Error inesperado al buscar eventos en el calendario {calendar_type}: {e}")
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
            params = {
                'calendarId': self.calendar_id,
                'timeMin': date_utils.format_date_for_api(start_date),
                'timeMax': date_utils.format_date_for_api(end_date),
                'singleEvents': True,
                'orderBy': 'startTime'
            }
            
            if max_results:
                params['maxResults'] = max_results
                
            events_result = self.service.events().list(**params).execute()
            events = events_result.get('items', [])
            logger.info(f"Se encontraron {len(events)} eventos en el rango de fechas")
            return self._format_events(events)
            
        except HttpError as error:
            logger.error(f"Error al buscar eventos por rango de fechas: {error}")
            return []

    def get_upcoming_events(self, max_results: int = 10) -> List[Dict]:
        """
        Obtiene los próximos eventos.
        Args:
            max_results: Número máximo de eventos a retornar
        """
        logger.info(f"Buscando los próximos {max_results} eventos")
        try:
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=datetime.now(datetime.UTC).isoformat() + 'Z',
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            logger.info(f"Se encontraron {len(events)} eventos próximos")
            return self._format_events(events)
            
        except HttpError as error:
            logger.error(f"Error al buscar eventos próximos: {error}")
            return []

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
                    start_formatted = start_dt.strftime('%A %d/%m/%Y')
                    end_formatted = end_dt.strftime('%A %d/%m/%Y')
                else:
                    # Convertir a zona horaria local
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    start_dt = start_dt.astimezone(self.timezone)
                    end_dt = end_dt.astimezone(self.timezone)
                    start_formatted = start_dt.strftime('%A %d/%m/%Y %H:%M')
                    end_formatted = end_dt.strftime('%A %d/%m/%Y %H:%M')
                
                formatted_events.append({
                    'summary': event.get('summary', 'Sin título'),
                    'start': start_formatted,
                    'end': end_formatted,
                    'description': event.get('description', ''),
                    'is_all_day': 'T' not in start,
                    'calendar_type': event.get('calendar_type', 'general')
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