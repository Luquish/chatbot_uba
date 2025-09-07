"""
Utilidades para el manejo de fechas en el sistema RAG.
Incluye funciones para calcular rangos de fechas, determinar el día de la semana,
formatear fechas y parsear fechas desde texto.
"""

import datetime
import logging
import re
import calendar
from typing import Tuple, Optional, List, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapeo de nombres de días de la semana en español
WEEKDAY_NAMES_ES = {
    0: 'lunes',
    1: 'martes',
    2: 'miércoles',
    3: 'jueves',
    4: 'viernes',
    5: 'sábado',
    6: 'domingo'
}

# Mapeo para referencias temporales relativas
RELATIVE_DAY_REFS = {
    'hoy': 0,
    'mañana': 1,
    'pasado mañana': 2,
    'ayer': -1,
    'anteayer': -2
}

class DateUtils:
    """Clase con utilidades para manejo de fechas"""

    def __init__(self):
        """Inicializa la clase de utilidades de fecha"""
        # Timezone por defecto para Argentina/Buenos Aires
        self.timezone_name = "America/Argentina/Buenos_Aires"

    def get_today(self) -> datetime.date:
        """
        Obtiene la fecha actual.
        
        Returns:
            datetime.date: Fecha actual
        """
        return datetime.date.today()
    
    def get_current_date(self) -> datetime.date:
        """
        Alias para get_today() por compatibilidad.
        
        Returns:
            datetime.date: Fecha actual
        """
        return self.get_today()

    def get_current_weekday(self) -> int:
        """
        Obtiene el día de la semana actual (0=Lunes, 6=Domingo).
        
        Returns:
            int: Número de día de la semana (0-6)
        """
        # En Python, weekday() retorna 0 para lunes, 6 para domingo
        return self.get_today().weekday()

    def get_current_weekday_name(self, locale: str = 'es') -> str:
        """
        Obtiene el nombre del día de la semana actual.
        
        Args:
            locale (str): Idioma ('es' para español, 'en' para inglés)
            
        Returns:
            str: Nombre del día de la semana
        """
        weekday = self.get_current_weekday()
        if locale.lower() == 'es':
            return WEEKDAY_NAMES_ES[weekday]
        else:
            # Usar nombres en inglés por defecto
            return calendar.day_name[weekday]

    def get_weekday_name(self, date: datetime.date, locale: str = 'es') -> str:
        """
        Obtiene el nombre del día de la semana para una fecha dada.
        
        Args:
            date (datetime.date): Fecha para la que obtener el nombre del día
            locale (str): Idioma ('es' para español, 'en' para inglés)
            
        Returns:
            str: Nombre del día de la semana
        """
        weekday = date.weekday()
        if locale.lower() == 'es':
            return WEEKDAY_NAMES_ES[weekday]
        else:
            return calendar.day_name[weekday]

    def get_this_week_date_range(self) -> Tuple[datetime.date, datetime.date]:
        """
        Obtiene el rango de fechas para la semana actual (lunes a domingo).
        
        Returns:
            Tuple[datetime.date, datetime.date]: (fecha_inicio, fecha_fin) de la semana
        """
        today = self.get_today()
        # Obtener el día de la semana (0=lunes, 6=domingo)
        weekday = today.weekday()
        # Calcular el lunes de esta semana
        monday = today - datetime.timedelta(days=weekday)
        # Calcular el domingo de esta semana
        sunday = monday + datetime.timedelta(days=6)
        return monday, sunday

    def get_formatted_date_range(self, start_date: datetime.date, end_date: datetime.date) -> str:
        """
        Formatea un rango de fechas para mostrar al usuario.
        
        Args:
            start_date (datetime.date): Fecha de inicio
            end_date (datetime.date): Fecha de fin
            
        Returns:
            str: Rango de fechas formateado (ej: "10 al 16 de mayo de 2025")
        """
        months_es = {
            1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
            5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
            9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
        }
        
        if start_date.month == end_date.month and start_date.year == end_date.year:
            return f"{start_date.day} al {end_date.day} de {months_es[start_date.month]} de {start_date.year}"
        elif start_date.year == end_date.year:
            return f"{start_date.day} de {months_es[start_date.month]} al {end_date.day} de {months_es[end_date.month]} de {start_date.year}"
        else:
            return f"{start_date.day} de {months_es[start_date.month]} de {start_date.year} al {end_date.day} de {months_es[end_date.month]} de {end_date.year}"

    def get_current_month_name(self, date: datetime.date = None) -> str:
        """
        Obtiene el nombre del mes actual en español y en mayúsculas.
        
        Args:
            date: Fecha específica, si no se proporciona usa la fecha actual
        
        Returns:
            str: Nombre del mes actual (ej: "AGOSTO", "SEPTIEMBRE")
        """
        months_es = {
            1: 'ENERO', 2: 'FEBRERO', 3: 'MARZO', 4: 'ABRIL',
            5: 'MAYO', 6: 'JUNIO', 7: 'JULIO', 8: 'AGOSTO',
            9: 'SEPTIEMBRE', 10: 'OCTUBRE', 11: 'NOVIEMBRE', 12: 'DICIEMBRE'
        }
        
        target_date = date if date else self.get_today()
        return months_es[target_date.month]
    
    def get_next_month_name(self, date: datetime.date = None) -> str:
        """
        Obtiene el nombre del próximo mes en español.
        
        Args:
            date: Fecha base, si no se proporciona usa la fecha actual
            
        Returns:
            str: Nombre del próximo mes (ej: "septiembre", "octubre")
        """
        months_es = {
            1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
            5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
            9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
        }
        
        target_date = date if date else self.get_today()
        next_month = target_date.month + 1
        next_year = target_date.year
        
        if next_month > 12:
            next_month = 1
            next_year += 1
            
        return months_es[next_month]

    # Nuevos helpers de normalización
    @staticmethod
    def month_name_to_num(name: str) -> Optional[int]:
        name = (name or "").strip().lower()
        months_es_to_num = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        return months_es_to_num.get(name)

    @staticmethod
    def num_to_month_name(num: int, uppercase: bool = False) -> str:
        months_es = {
            1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
            5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
            9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
        }
        name = months_es.get(num, '')
        return name.upper() if uppercase else name

    @staticmethod
    def detect_month_from_text(text: str) -> Optional[str]:
        """
        Devuelve el nombre de mes (minúsculas) si aparece en el texto.
        """
        text = (text or "").lower()
        for m in ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre']:
            if m in text:
                return m
        return None

    def parse_date_from_text(self, text: str) -> Optional[datetime.date]:
        """
        Intenta extraer una fecha específica de un texto.
        
        Args:
            text (str): Texto que puede contener una fecha
            
        Returns:
            Optional[datetime.date]: Fecha extraída o None si no se encontró
        """
        today = self.get_today()
        
        # Buscar referencias a días relativos (hoy, mañana, etc.)
        text_lower = text.lower()
        for day_ref, days_delta in RELATIVE_DAY_REFS.items():
            if day_ref in text_lower:
                return today + datetime.timedelta(days=days_delta)
        
        # Buscar referencias a días de la semana (lunes, martes, etc.)
        current_weekday = today.weekday()
        
        # Crear un diccionario inverso para buscar por nombre del día
        weekday_indices = {name: idx for idx, name in WEEKDAY_NAMES_ES.items()}
        
        for day_name, day_idx in weekday_indices.items():
            if day_name in text_lower:
                # Calcular cuántos días hay que avanzar para llegar al día mencionado
                days_ahead = (day_idx - current_weekday) % 7
                if days_ahead == 0:
                    # Si es el mismo día de la semana actual, verificar si se refiere al próximo
                    if any(word in text_lower for word in ['próximo', 'próxima', 'siguiente', 'que viene']):
                        days_ahead = 7
                return today + datetime.timedelta(days=days_ahead)
        
        # Intentar encontrar fechas en formatos comunes (DD/MM, DD/MM/YYYY)
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?',  # DD/MM/YYYY o DD/MM
            r'(\d{1,2})-(\d{1,2})(?:-(\d{2,4}))?',  # DD-MM-YYYY o DD-MM
            r'(\d{1,2}) de (\w+)(?: de (\d{2,4}))?'  # DD de Mes de YYYY o DD de Mes
        ]
        
        months_es_to_num = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                day = int(match.group(1))
                # Segundo grupo puede ser número de mes o nombre de mes
                if match.group(2).isdigit():
                    month = int(match.group(2))
                else:
                    # Buscar nombre de mes en formato texto
                    month_name = match.group(2)
                    if month_name in months_es_to_num:
                        month = months_es_to_num[month_name]
                    else:
                        # Intentar con coincidencias parciales
                        for month_full, month_num in months_es_to_num.items():
                            if month_name in month_full:
                                month = month_num
                                break
                        else:
                            continue
                
                # Manejar año (si está presente)
                if match.group(3):
                    year = int(match.group(3))
                    # Ajustar formato de 2 dígitos a 4 dígitos
                    if year < 100:
                        year += 2000
                else:
                    year = today.year
                
                try:
                    return datetime.date(year, month, day)
                except ValueError:
                    # Fecha inválida (por ejemplo, 31 de febrero)
                    continue
        
        return None

    def get_dates_for_weekday(self, weekday_name: str, within_days: int = 30) -> List[datetime.date]:
        """
        Obtiene las próximas fechas para un día de la semana específico.
        
        Args:
            weekday_name (str): Nombre del día de la semana en español
            within_days (int): Número de días a considerar en adelante
            
        Returns:
            List[datetime.date]: Lista de fechas que corresponden al día de semana solicitado
        """
        weekday_name = weekday_name.lower()
        
        # Encontrar el índice del día de la semana
        weekday_idx = -1
        for idx, name in WEEKDAY_NAMES_ES.items():
            if name == weekday_name:
                weekday_idx = idx
                break
        
        if weekday_idx == -1:
            return []
        
        today = self.get_today()
        result_dates = []
        
        for i in range(within_days):
            check_date = today + datetime.timedelta(days=i)
            if check_date.weekday() == weekday_idx:
                result_dates.append(check_date)
        
        return result_dates

    def extract_date_range_from_query(self, query: str) -> Tuple[Optional[datetime.date], Optional[datetime.date]]:
        """
        Extrae rango de fechas de una consulta (ej: "esta semana", "próximo mes").
        
        Args:
            query (str): Consulta del usuario
            
        Returns:
            Tuple[Optional[datetime.date], Optional[datetime.date]]: (fecha_inicio, fecha_fin)
        """
        query = query.lower()
        today = self.get_today()
        
        # Comprobar si se refiere a esta semana
        if any(term in query for term in ['esta semana', 'semana actual', 'en la semana']):
            return self.get_this_week_date_range()
            
        # Comprobar si se refiere a la próxima semana
        elif any(term in query for term in ['proxima semana', 'próxima semana', 'semana que viene', 'semana siguiente']):
            next_week_monday = today + datetime.timedelta(days=(7 - today.weekday()))
            next_week_sunday = next_week_monday + datetime.timedelta(days=6)
            return next_week_monday, next_week_sunday
            
        # Hoy
        elif any(term in query for term in ['hoy', 'este dia', 'este día']):
            return today, today
            
        # Mañana
        elif any(term in query for term in ['mañana', 'manana']):
            tomorrow = today + datetime.timedelta(days=1)
            return tomorrow, tomorrow
            
        # Día específico de la semana
        for day_name, day_idx in WEEKDAY_NAMES_ES.items():
            if day_name in query:
                # Calcular días hasta ese día de la semana
                days_ahead = (day_idx - today.weekday()) % 7
                if days_ahead == 0:
                    # Si estamos en ese día, verificar si se refiere al próximo
                    if any(word in query for word in ['próximo', 'próxima', 'siguiente', 'que viene']):
                        days_ahead = 7
                target_date = today + datetime.timedelta(days=days_ahead)
                return target_date, target_date
            
        return None, None

    def get_formatted_date(self, date: datetime.date) -> str:
        """
        Formatea una fecha para mostrar al usuario.
        
        Args:
            date (datetime.date): Fecha a formatear
            
        Returns:
            str: Fecha formateada (ej: "10 de mayo de 2025")
        """
        months_es = {
            1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
            5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
            9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
        }
        return f"{date.day} de {months_es[date.month]} de {date.year}"

    @staticmethod
    def get_weekday_abbr(name: str) -> str:
        """
        Devuelve abreviatura en español para un nombre de día (ej.: miércoles→mié).
        Acepta variantes sin tilde.
        """
        name = (name or '').lower()
        mapping = {
            'lunes': 'lun', 'martes': 'mar', 'miércoles': 'mié', 'miercoles': 'mié',
            'jueves': 'jue', 'viernes': 'vie', 'sábado': 'sáb', 'sabado': 'sáb', 'domingo': 'dom'
        }
        return mapping.get(name, '')

    def format_date_for_api(self, date: datetime.date) -> str:
        """
        Formatea una fecha para usar en APIs.
        
        Args:
            date (datetime.date): Fecha a formatear
            
        Returns:
            str: Fecha en formato ISO 8601
        """
        return date.isoformat() + "T00:00:00Z"

    def get_courses_for_date(self, courses: List[Dict[str, Any]], target_date: datetime.date) -> List[Dict[str, Any]]:
        """
        Filtra cursos para una fecha específica.
        
        Args:
            courses (List[Dict[str, Any]]): Lista de cursos con información de fecha
            target_date (datetime.date): Fecha objetivo
            
        Returns:
            List[Dict[str, Any]]: Cursos filtrados para la fecha
        """
        matching_courses = []
        
        # Formato esperado en la hoja: DD/MM
        target_date_str = f"{target_date.day:02d}/{target_date.month:02d}"
        
        for course in courses:
            # Verificar que el curso tenga campo fecha y comparar
            if 'fecha' in course and course['fecha']:
                if course['fecha'] == target_date_str:
                    matching_courses.append(course)
        
        return matching_courses

    def get_courses_for_weekday(self, courses: List[Dict[str, Any]], weekday_name: str) -> List[Dict[str, Any]]:
        """
        Filtra cursos para un día de la semana específico.
        
        Args:
            courses (List[Dict[str, Any]]): Lista de cursos con información de fecha
            weekday_name (str): Nombre del día de la semana en español
            
        Returns:
            List[Dict[str, Any]]: Cursos filtrados para el día de la semana
        """
        # Encontrar el índice del día de la semana
        weekday_idx = -1
        weekday_name = weekday_name.lower()
        
        for idx, name in WEEKDAY_NAMES_ES.items():
            if name == weekday_name:
                weekday_idx = idx
                break
                
        if weekday_idx == -1:
            return []
            
        matching_courses = []
        today = self.get_today()
        
        # Buscar las próximas 4 instancias de ese día de la semana
        for i in range(28):  # 4 semanas
            check_date = today + datetime.timedelta(days=i)
            if check_date.weekday() == weekday_idx:
                date_str = f"{check_date.day:02d}/{check_date.month:02d}"
                for course in courses:
                    if 'fecha' in course and course['fecha'] == date_str:
                        matching_courses.append(course)
                        
        return matching_courses

    def get_courses_for_this_week(self, courses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtra cursos para la semana actual.
        
        Args:
            courses (List[Dict[str, Any]]): Lista de cursos con información de fecha
            
        Returns:
            List[Dict[str, Any]]: Cursos filtrados para esta semana
        """
        monday, sunday = self.get_this_week_date_range()
        
        matching_courses = []
        current_year = self.get_today().year
        
        for course in courses:
            if 'fecha' not in course or not course['fecha']:
                continue
                
            # Intentar parsear la fecha del curso (generalmente en formato DD/MM)
            try:
                parts = course['fecha'].split('/')
                if len(parts) >= 2:
                    day = int(parts[0])
                    month = int(parts[1])
                    
                    # Asumir el año actual
                    course_date = datetime.date(current_year, month, day)
                    
                    # Verificar si la fecha está en el rango de esta semana
                    if monday <= course_date <= sunday:
                        matching_courses.append(course)
            except (ValueError, IndexError):
                # Si hay un error al parsear la fecha, ignorar este curso
                continue
                
        return matching_courses

    def add_days(self, date: datetime.date, days: int) -> datetime.date:
        """
        Añade un número determinado de días a una fecha.
        
        Args:
            date (datetime.date): Fecha base
            days (int): Número de días a añadir (puede ser negativo)
            
        Returns:
            datetime.date: Nueva fecha resultante
        """
        return date + datetime.timedelta(days=days)

    def get_date(self, year: int, month: int, day: int) -> datetime.date:
        """
        Crea un objeto datetime.date con los valores proporcionados.
        
        Args:
            year (int): Año
            month (int): Mes (1-12)
            day (int): Día (1-31)
            
        Returns:
            datetime.date: Objeto de fecha
        """
        return datetime.date(year, month, day)

if __name__ == "__main__":
    # Pruebas básicas
    date_utils = DateUtils()
    
    print(f"Hoy es: {date_utils.get_today()}")
    print(f"Día de la semana actual: {date_utils.get_current_weekday_name()}")
    
    monday, sunday = date_utils.get_this_week_date_range()
    print(f"Esta semana va del {monday} al {sunday}")
    print(f"Formatted: {date_utils.get_formatted_date_range(monday, sunday)}")
    
    # Prueba de extracción de fechas de texto
    texts = [
        "¿Qué hay para mañana?",
        "¿Cuáles son los cursos del próximo lunes?",
        "Necesito información sobre el curso del 15/05",
        "¿Hay algo este viernes?",
        "¿Qué pasa el 20 de mayo de 2025?"
    ]
    
    for text in texts:
        date = date_utils.parse_date_from_text(text)
        if date:
            print(f"De '{text}' -> {date} ({date_utils.get_weekday_name(date)})")
        else:
            print(f"No se pudo extraer fecha de '{text}'") 