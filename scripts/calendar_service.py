"""
Servicio para interactuar con Google Calendar API.
Solo lectura de eventos públicos usando API Key.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pytz
from config.calendar_config import CALENDAR_CONFIG, DATE_FORMAT

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalendarService:
    def __init__(self):
        """Inicializa el servicio de Google Calendar."""
        if not CALENDAR_CONFIG['GOOGLE_CALENDAR_ID'] or not CALENDAR_CONFIG['GOOGLE_CALENDAR_API_KEY']:
            raise ValueError("GOOGLE_CALENDAR_ID y GOOGLE_CALENDAR_API_KEY son requeridos en la configuración")
            
        try:
            self.service = build('calendar', 'v3', developerKey=CALENDAR_CONFIG['GOOGLE_CALENDAR_API_KEY'])
            self.timezone = pytz.timezone(CALENDAR_CONFIG['TIMEZONE'])
            logger.info("Servicio de Google Calendar inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el servicio de Google Calendar: {str(e)}")
            raise

    def get_events_this_week(self) -> List[Dict]:
        """
        Obtiene los eventos de la semana actual.
        
        Returns:
            List[Dict]: Lista de eventos con formato simplificado
        """
        try:
            # Log de configuración
            logger.info(f"Intentando obtener eventos con Calendar ID: {CALENDAR_CONFIG['GOOGLE_CALENDAR_ID']}")
            
            # Calcular inicio y fin de la semana en la zona horaria correcta
            now = datetime.now(self.timezone)
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
            
            # Convertir a UTC para la API
            time_min = start.astimezone(pytz.UTC).isoformat()
            time_max = end.astimezone(pytz.UTC).isoformat()
            
            logger.info(f"Buscando eventos entre {time_min} y {time_max}")
            
            # Hacer la llamada a la API
            events_result = self.service.events().list(
                calendarId=CALENDAR_CONFIG['GOOGLE_CALENDAR_ID'],
                timeMin=time_min,
                timeMax=time_max,
                maxResults=CALENDAR_CONFIG['MAX_RESULTS'],
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            logger.info(f"Se encontraron {len(events)} eventos en el calendario")
            
            if not events:
                logger.info("No se encontraron eventos para esta semana")
                return []
                
            # Formatear eventos
            formatted_events = []
            for event in events:
                # Obtener fechas y convertir a zona horaria local
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                # Si es un evento de todo el día (solo fecha)
                if 'T' not in start:  # No tiene componente de tiempo
                    start_dt = datetime.strptime(start, '%Y-%m-%d')
                    end_dt = datetime.strptime(end, '%Y-%m-%d')
                    date_format = DATE_FORMAT['DATE']
                else:
                    # Convertir a zona horaria local
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    start_dt = start_dt.astimezone(self.timezone)
                    end_dt = end_dt.astimezone(self.timezone)
                    date_format = DATE_FORMAT['DATETIME']
                
                formatted_events.append({
                    'summary': event.get('summary', 'Sin título'),
                    'start': start_dt.strftime(date_format),
                    'end': end_dt.strftime(date_format),
                    'description': event.get('description', ''),
                    'is_all_day': 'T' not in start
                })
                
            logger.info(f"Se encontraron {len(formatted_events)} eventos")
            return formatted_events
            
        except HttpError as error:
            logger.error(f"Error al obtener eventos: {str(error)}")
            return []
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            return []

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