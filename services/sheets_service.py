"""
Servicio para interactuar con Google Sheets API.
Permite leer datos de hojas de cálculo públicas usando una API Key.
"""

import os
import logging
from typing import List, Dict, Optional, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sys
from datetime import datetime, timedelta

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar configuración desde variables de entorno
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# El usuario debe configurar esta variable de entorno con el ID de su Google Sheet
CURSOS_SPREADSHEET_ID = os.getenv('CURSOS_SPREADSHEET_ID')

class SheetsService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el servicio de Google Sheets.

        Args:
            api_key (Optional[str]): La API key para Google Sheets. Si no se provee,
                                     se tomará de la variable de entorno GOOGLE_API_KEY.
        """
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Se requiere GOOGLE_API_KEY en las variables de entorno o como argumento.")

        try:
            self.service = build('sheets', 'v4',
                                 developerKey=self.api_key,
                                 cache_discovery=False)
            logger.info("Servicio de Google Sheets inicializado correctamente.")
        except Exception as e:
            logger.error(f"Error al inicializar el servicio de Google Sheets: {e}")
            raise

    def get_sheet_data(self, spreadsheet_id: str, ranges: List[str], include_grid_data: bool = False) -> Optional[Dict]:
        """
        Obtiene datos de una hoja de cálculo específica y rangos.

        Args:
            spreadsheet_id (str): El ID de la hoja de cálculo.
            ranges (List[str]): Lista de rangos a leer en notación A1 (ej: ["'Sheet Name'!A1:B2", "'Other Sheet'!C3:D4"]).
                                Es importante encerrar los nombres de hoja con espacios o caracteres especiales entre comillas simples.
            include_grid_data (bool): Si es True, incluye todos los datos de la cuadrícula.

        Returns:
            Optional[Dict]: Los datos de la hoja de cálculo o None si ocurre un error.
        """
        try:
            logger.info(f"Solicitando datos de la hoja de cálculo {spreadsheet_id} para los rangos: {ranges}")
            request = self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id,
                ranges=ranges,
                includeGridData=include_grid_data
            )
            response = request.execute()
            logger.info(f"Datos obtenidos exitosamente de {spreadsheet_id}.")
            return response
        except HttpError as error:
            logger.error(f"Error de API al obtener datos de Google Sheets ({spreadsheet_id}): {error.resp.status} - {error._get_reason()}")
            logger.error(f"Detalles del error: {error.content}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado al obtener datos de Google Sheets ({spreadsheet_id}): {e}")
            return None

    def get_sheet_values(self, spreadsheet_id: str, range_name: str) -> Optional[List[List[Any]]]:
        """
        Obtiene los valores de un rango específico de una hoja de cálculo.

        Args:
            spreadsheet_id (str): El ID de la hoja de cálculo.
            range_name (str): El rango a leer en notación A1 (ej: "'Sheet Name'!A1:B2").
                              Es importante encerrar los nombres de hoja con espacios o caracteres especiales entre comillas simples.

        Returns:
            Optional[List[List[Any]]]: Lista de listas representando las filas y valores, o None si ocurre un error.
                                       Devuelve una lista vacía si el rango es válido pero no contiene datos.
        """
        try:
            logger.info(f"Solicitando valores de la hoja de cálculo {spreadsheet_id} para el rango: {range_name}")
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            values = result.get('values', [])
            logger.info(f"Valores obtenidos exitosamente de {spreadsheet_id} para el rango {range_name}. Total filas: {len(values)}")
            return values
        except HttpError as error:
            logger.error(f"Error de API al obtener valores de Google Sheets ({spreadsheet_id}, {range_name}): {error.resp.status} - {error._get_reason()}")
            logger.error(f"Detalles del error: {error.content}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado al obtener valores de Google Sheets ({spreadsheet_id}, {range_name}): {e}")
            return None

    def get_courses_by_date(self, spreadsheet_id: str, sheet_name: str, date_str: str, date_col_index: int = 0, date_format: str = "%d/%m") -> Optional[List[Dict[str, Any]]]:
        """
        Obtiene todos los cursos para una fecha específica.

        Args:
            spreadsheet_id (str): ID de la hoja de cálculo
            sheet_name (str): Nombre de la hoja (ej: "MARZO")
            date_str (str): Fecha en formato DD/MM
            date_col_index (int): Índice de la columna donde se encuentra la fecha (0 = columna A)
            date_format (str): Formato de fecha usado en la hoja

        Returns:
            Optional[List[Dict[str, Any]]]: Lista de cursos para esa fecha o None si hay error
        """
        try:
            # Primero obtener todas las filas
            range_name = f"'{sheet_name}'!A:E"  # Asumimos columnas A-E para datos de cursos
            values = self.get_sheet_values(spreadsheet_id, range_name)
            
            if not values:
                logger.info(f"No se encontraron datos en la hoja {sheet_name}")
                return []
            
            # La primera fila contiene los encabezados
            headers = values[0]
            
            # Filtrar filas que coinciden con la fecha
            filtered_courses = []
            for i, row in enumerate(values[1:], 1):  # Saltamos la fila de encabezado
                # Verificar si hay suficientes columnas y si la fecha coincide
                if len(row) > date_col_index and row[date_col_index] == date_str:
                    course = {headers[j]: row[j] if j < len(row) else "" for j in range(len(headers))}
                    filtered_courses.append(course)
            
            logger.info(f"Se encontraron {len(filtered_courses)} cursos para la fecha {date_str}")
            return filtered_courses
        
        except Exception as e:
            logger.error(f"Error al obtener cursos por fecha {date_str}: {e}")
            return None

    def get_courses_this_week(self, spreadsheet_id: str, sheet_name: str, date_col_index: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Obtiene todos los cursos programados para la semana actual.

        Args:
            spreadsheet_id (str): ID de la hoja de cálculo
            sheet_name (str): Nombre de la hoja (ej: "MARZO")
            date_col_index (int): Índice de la columna donde se encuentra la fecha (0 = columna A)

        Returns:
            Optional[List[Dict[str, Any]]]: Lista de cursos para esta semana o None si hay error
        """
        try:
            # Importamos aquí para evitar dependencia circular
            from utils.date_utils import DateUtils
            date_utils = DateUtils()
            
            # Obtener fechas de esta semana
            monday, sunday = date_utils.get_this_week_date_range()
            current_date = monday
            
            # Crear lista de fechas en formato DD/MM
            dates_this_week = []
            while current_date <= sunday:
                date_str = f"{current_date.day:02d}/{current_date.month:02d}"
                dates_this_week.append(date_str)
                current_date += datetime.timedelta(days=1)
            
            # Obtener todos los cursos
            range_name = f"'{sheet_name}'!A:E"
            values = self.get_sheet_values(spreadsheet_id, range_name)
            
            if not values:
                logger.info(f"No se encontraron datos en la hoja {sheet_name}")
                return []
            
            # La primera fila contiene los encabezados
            headers = values[0]
            
            # Filtrar cursos que coinciden con las fechas de esta semana
            weekly_courses = []
            for i, row in enumerate(values[1:], 1):  # Saltamos la fila de encabezado
                if len(row) > date_col_index and row[date_col_index] in dates_this_week:
                    course = {headers[j]: row[j] if j < len(row) else "" for j in range(len(headers))}
                    weekly_courses.append(course)
            
            logger.info(f"Se encontraron {len(weekly_courses)} cursos para esta semana")
            return weekly_courses
        
        except Exception as e:
            logger.error(f"Error al obtener cursos de esta semana: {e}")
            return None

    def get_course_by_weekday_and_name(self, spreadsheet_id: str, sheet_name: str, weekday_name: str, 
                                      course_name: str = None, date_col_index: int = 0, 
                                      name_col_index: int = 2) -> Optional[Dict[str, Any]]:
        """
        Busca un curso específico por día de la semana y nombre (si se proporciona).

        Args:
            spreadsheet_id (str): ID de la hoja de cálculo
            sheet_name (str): Nombre de la hoja
            weekday_name (str): Nombre del día de la semana en español (ej: "lunes")
            course_name (str, opcional): Nombre del curso para filtrar
            date_col_index (int): Índice de la columna de fecha
            name_col_index (int): Índice de la columna de nombre de curso

        Returns:
            Optional[Dict[str, Any]]: Datos del curso o None si no se encuentra
        """
        try:
            # Importamos aquí para evitar dependencia circular
            from utils.date_utils import DateUtils
            date_utils = DateUtils()
            
            # Obtener próximas fechas para ese día de la semana
            weekday_dates = date_utils.get_dates_for_weekday(weekday_name)
            if not weekday_dates:
                logger.warning(f"No se encontraron fechas próximas para {weekday_name}")
                return None
            
            date_strings = [f"{date.day:02d}/{date.month:02d}" for date in weekday_dates]
            
            # Obtener todos los cursos
            range_name = f"'{sheet_name}'!A:E"
            values = self.get_sheet_values(spreadsheet_id, range_name)
            
            if not values:
                logger.info(f"No se encontraron datos en la hoja {sheet_name}")
                return None
            
            # La primera fila contiene los encabezados
            headers = values[0]
            
            # Buscar cursos que coinciden con el día de la semana y opcionalmente el nombre
            for i, row in enumerate(values[1:], 1):
                if len(row) > date_col_index and row[date_col_index] in date_strings:
                    # Si no se especificó nombre, o si el nombre coincide
                    if not course_name or (len(row) > name_col_index and course_name.lower() in row[name_col_index].lower()):
                        course = {headers[j]: row[j] if j < len(row) else "" for j in range(len(headers))}
                        return course
            
            logger.info(f"No se encontró ningún curso para {weekday_name} con el nombre {course_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error al buscar curso por día de la semana: {e}")
            return None

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
    elif not CURSOS_SPREADSHEET_ID:
        print("Error: La variable de entorno CURSOS_SPREADSHEET_ID no está configurada.")
    else:
        try:
            sheets_service = SheetsService()
            spreadsheet_id_to_test = CURSOS_SPREADSHEET_ID
            
            # Ajusta el nombre de la hoja y el rango según tu Google Sheet.
            # La imagen muestra una pestaña llamada "MARZO" y columnas hasta la E.
            # Filas A1:E11 leería las cabeceras y 10 filas de datos.
            test_sheet_name = "MARZO" # O el nombre de la hoja que contiene los datos principales
            range_to_fetch_detailed = [f"'{test_sheet_name}'!A1:E11"]
            range_to_fetch_values = f"'{test_sheet_name}'!A1:E"

            print(f"\nProbando get_sheet_data para {spreadsheet_id_to_test} con rangos {range_to_fetch_detailed}...")
            data = sheets_service.get_sheet_data(spreadsheet_id_to_test, range_to_fetch_detailed, include_grid_data=True)
            
            if data:
                print("Respuesta (resumida) de get_sheet_data:")
                for sheet_info in data.get('sheets', []):
                    print(f"  Procesando hoja: {sheet_info.get('properties', {}).get('title')}")
                    for grid_data_block in sheet_info.get('data', []):
                        row_data_list = grid_data_block.get('rowData', [])
                        print(f"    Datos del rango recuperado (primeras {min(5, len(row_data_list))} filas):")
                        for i, row_data in enumerate(row_data_list[:5]):
                            formatted_values = [cell.get('formattedValue', '') for cell in row_data.get('values', []) if cell]
                            print(f"      Fila {grid_data_block.get('startRow', 0) + i + 1}: {formatted_values}")
                        if not row_data_list:
                            print("    No se encontró rowData para este bloque.")
            else:
                print(f"No se pudieron obtener datos con get_sheet_data para {spreadsheet_id_to_test}")

            print(f"\nProbando get_sheet_values para {spreadsheet_id_to_test} con rango {range_to_fetch_values}...")
            values = sheets_service.get_sheet_values(spreadsheet_id_to_test, range_to_fetch_values)

            if values:
                print(f"Valores obtenidos de {range_to_fetch_values} (primeras 10 filas):")
                for i, row in enumerate(values[:10]):
                    print(f"  Fila {i+1}: {row}")
            elif values == []:
                print(f"El rango {range_to_fetch_values} es válido pero no contiene valores o todas las celdas están vacías.")
            else:
                print(f"No se pudieron obtener valores con get_sheet_values para {spreadsheet_id_to_test} y rango {range_to_fetch_values}.")

        except ValueError as ve:
            print(f"Error de configuración o inicialización: {ve}")
        except Exception as e:
            print(f"Ocurrió un error durante las pruebas: {e}")
