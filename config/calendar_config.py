"""
Configuración específica para la lectura del calendario académico de la Facultad de Medicina UBA usando Pydantic.
"""

import os
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class CalendarInfo(BaseSettings):
    """Información específica de un calendario."""
    
    calendar_id: Optional[str] = Field(default=None, description="ID del calendario")
    name: str = Field(..., description="Nombre del calendario")
    description: str = Field(..., description="Descripción del calendario")
    keywords: List[str] = Field(..., description="Palabras clave asociadas")
    color: str = Field(..., description="Color del calendario")

    class Config:
        env_prefix = ''


class CalendarSettings(BaseSettings):
    """Configuración de un único calendario de actividades CECIM."""
    
    calendar_id_actividades_cecim: Optional[str] = Field(default=None, env='CALENDAR_ID_ACTIVIDADES_CECIM')
    
    class Config:
        env_prefix = ''
    
    def get_calendars_config(self) -> Dict[str, CalendarInfo]:
        """Retorna configuración única de calendario de actividades CECIM."""
        return {
            'actividades_cecim': CalendarInfo(
                calendar_id=self.calendar_id_actividades_cecim,
                name='ACTIVIDADES CECIM',
                description='Calendario unificado de actividades del CECIM',
                keywords=['examen', 'parcial', 'final', 'inscripcion', 'inscripción', 'cursada', 'actividad', 'evento'],
                color='#0b8043'
            )
        }


class CalendarApiSettings(BaseSettings):
    """Configuración general del API del calendario."""
    
    google_api_key: Optional[str] = Field(default=None, env='GOOGLE_API_KEY')
    base_url: str = Field(default='https://www.googleapis.com/calendar/v3')
    max_results: int = Field(default=10)
    timezone: str = Field(default='America/Argentina/Buenos_Aires')
    
    class Config:
        env_prefix = ''


class CalendarConfig(BaseSettings):
    """Configuración principal del sistema de calendarios."""
    
    calendar_settings: CalendarSettings = CalendarSettings()
    api_settings: CalendarApiSettings = CalendarApiSettings()
    
    class Config:
        env_prefix = ''

    def get_legacy_calendars_dict(self) -> Dict[str, Dict[str, any]]:
        """Retorna los calendarios en el formato legacy para compatibilidad."""
        calendars_config = self.calendar_settings.get_calendars_config()
        legacy_dict = {}
        
        for key, calendar_info in calendars_config.items():
            legacy_dict[key] = {
                'id': calendar_info.calendar_id,
                'name': calendar_info.name,
                'description': calendar_info.description,
                'keywords': calendar_info.keywords,
                'color': calendar_info.color
            }
        
        return legacy_dict

    def get_legacy_config_dict(self) -> Dict[str, any]:
        """Retorna la configuración en el formato legacy para compatibilidad."""
        return {
            'GOOGLE_API_KEY': self.api_settings.google_api_key,
            'BASE_URL': self.api_settings.base_url,
            'MAX_RESULTS': self.api_settings.max_results,
            'TIMEZONE': self.api_settings.timezone
        }


# =============================================================================
# INSTANCIA GLOBAL Y COMPATIBILIDAD LEGACY
# =============================================================================

# Crear instancia global
calendar_config = CalendarConfig()

# Variables legacy para compatibilidad
CALENDARS = calendar_config.get_legacy_calendars_dict()
CALENDAR_CONFIG = calendar_config.get_legacy_config_dict()

# Acceso directo a la configuración estructurada (nueva funcionalidad)
calendars_info = calendar_config.calendar_settings.get_calendars_config()
api_config = calendar_config.api_settings