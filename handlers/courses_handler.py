"""
Manejador para consultas sobre cursos desde Google Sheets.
"""
import logging
import random
import datetime
from typing import List, Dict, Any, Optional

from config.constants import SHEET_QUERY_CONFIG, SHEET_COURSE_KEYWORDS, information_emojis
from config.settings import config
from services.sheets_service import SheetsService
from utils.date_utils import DateUtils

logger = logging.getLogger(__name__)


def parse_sheet_course_data(rows: List[List[str]]) -> List[Dict]:
    """
    Convierte filas de datos de cursos de la hoja de cálculo a una lista de diccionarios.
    Asume que la primera fila de `rows` son las cabeceras.
    
    Args:
        rows (List[List[str]]): Filas de datos de la hoja de cálculo
        
    Returns:
        List[Dict]: Lista de cursos como diccionarios
    """
    if not rows or len(rows) < SHEET_QUERY_CONFIG['header_row'] + 1: # Necesita cabecera y al menos una fila de datos
        logger.warning("No hay suficientes filas para parsear (se necesita cabecera y datos).")
        return []
    
    header_idx = SHEET_QUERY_CONFIG['header_row'] - 1
    headers_from_sheet = [str(header).strip() for header in rows[header_idx]]
    
    # Mapear nombres de columna de la configuración a índices reales basados en las cabeceras leídas
    col_map = {}
    
    # Mapeo flexible para diferentes nombres de columnas
    activity_aliases = ['NOMBRE DE ACTIVIDAD', 'ACTIVIDAD', 'NOMBRE', 'CURSO', 'NOMBRE DEL CURSO']
    form_aliases = ['FORMULARIO', 'LINK', 'ENLACE', 'URL']
    date_aliases = ['FECHA', 'DÍA', 'DIA']
    
    # Buscar columna de actividad
    col_map['activity'] = -1
    for alias in activity_aliases:
        if alias in headers_from_sheet:
            col_map['activity'] = headers_from_sheet.index(alias)
            logger.info(f"Columna de actividad encontrada: {alias}")
            break
    
    # Buscar columna de formulario
    col_map['form'] = -1
    for alias in form_aliases:
        if alias in headers_from_sheet:
            col_map['form'] = headers_from_sheet.index(alias)
            logger.info(f"Columna de formulario encontrada: {alias}")
            break
    
    # Buscar columna de fecha
    col_map['date'] = -1
    for alias in date_aliases:
        if alias in headers_from_sheet:
            col_map['date'] = headers_from_sheet.index(alias)
            logger.info(f"Columna de fecha encontrada: {alias}")
            break
    
    # Verificar que se encontraron las columnas esenciales
    if col_map['activity'] == -1 or col_map['date'] == -1:
        logger.error(f"No se pudieron encontrar las columnas esenciales. Cabeceras encontradas: {headers_from_sheet}")
        logger.error(f"Actividad: {col_map['activity']}, Fecha: {col_map['date']}")
        return []
    
    # Opcional: Mapear otras columnas si son necesarias (ej. HORARIO, MODALIDAD)
    col_map['time'] = headers_from_sheet.index('HORARIO') if 'HORARIO' in headers_from_sheet else -1
    col_map['modality'] = headers_from_sheet.index('MODALIDAD') if 'MODALIDAD' in headers_from_sheet else -1

    parsed_data = []
    for i, row in enumerate(rows[SHEET_QUERY_CONFIG['header_row']:]):
        # Saltar filas que no parezcan tener datos de curso (ej. las que dicen "SEMANA DEL...")
        # o filas completamente vacías.
        if not row or not any(field.strip() for field in row) or \
           (len(row) > col_map['activity'] and "SEMANA DEL" in row[col_map['activity']]):
            continue

        record = {}
        try:
            record['nombre_actividad'] = row[col_map['activity']].strip() if col_map['activity'] < len(row) else ""
            record['formulario'] = row[col_map['form']].strip() if col_map['form'] < len(row) else ""
            record['fecha'] = row[col_map['date']].strip() if col_map['date'] < len(row) else ""
            if col_map['time'] != -1:
                 record['horario'] = row[col_map['time']].strip() if col_map['time'] < len(row) else ""
            if col_map['modality'] != -1:
                record['modalidad'] = row[col_map['modality']].strip() if col_map['modality'] < len(row) else ""
            
            # Solo añadir si tiene nombre de actividad
            if record['nombre_actividad']:
                parsed_data.append(record)
        except IndexError:
            logger.warning(f"Fila {i + SHEET_QUERY_CONFIG['header_row'] + 1} con menos columnas de las esperadas. Fila: {row}")
            continue
    
    logger.info(f"Parseados {len(parsed_data)} cursos de la hoja.")
    return parsed_data


def format_course_info_for_response(course_data: Dict) -> str:
    """
    Formatea la información de un curso para la respuesta al usuario.
    
    Args:
        course_data (Dict): Datos del curso
        
    Returns:
        str: Texto formateado para la respuesta
    """
    response_parts = []
    if course_data.get('nombre_actividad'):
        response_parts.append(f"🎓 Curso: {course_data['nombre_actividad']}")
    if course_data.get('fecha'):
        response_parts.append(f"  🗓️ Fecha: {course_data['fecha']}")
    if course_data.get('horario'):
        response_parts.append(f"  ⏰ Horario: {course_data['horario']}")
    if course_data.get('formulario') and course_data['formulario'].startswith('http'):
        response_parts.append(f"  🔗 Formulario: {course_data['formulario']}")
    elif course_data.get('formulario'): # Si hay algo pero no es link
         response_parts.append(f"  📝 Formulario (info): {course_data['formulario']}")
    return "\n".join(response_parts)


def sort_courses_by_proximity(courses: List[Dict[str, Any]], date_utils: DateUtils) -> List[Dict[str, Any]]:
    """
    Ordena los cursos por proximidad a la fecha actual.
    
    Args:
        courses (List[Dict[str, Any]]): Lista de cursos
        date_utils (DateUtils): Utilidades de fecha
        
    Returns:
        List[Dict[str, Any]]: Cursos ordenados por proximidad
    """
    today = date_utils.get_today()
    current_year = today.year
    
    def get_date_distance(course):
        """Calcula la distancia en días desde hoy a la fecha del curso"""
        try:
            if 'fecha' not in course or not course['fecha']:
                return 999  # Cursos sin fecha van al final
            
            parts = course['fecha'].split('/')
            if len(parts) >= 2:
                day = int(parts[0])
                month = int(parts[1])
                
                # Crear la fecha y calcular la diferencia en días
                course_date = date_utils.get_date(current_year, month, day)
                
                # Si la fecha ya pasó este año, asumir que es para el próximo año
                if course_date < today:
                    course_date = date_utils.get_date(current_year + 1, month, day)
                
                return (course_date - today).days
        except (ValueError, IndexError):
            return 999
            
        return 999
    
    # Ordenar los cursos por distancia en días
    return sorted(courses, key=get_date_distance)


def handle_sheet_course_query(query: str, sheets_service: SheetsService, 
                             spreadsheet_id: str, date_utils: DateUtils) -> Optional[str]:
    """
    Maneja consultas específicas sobre cursos obteniendo datos de Google Sheets.
    
    Args:
        query (str): Consulta del usuario
        sheets_service (SheetsService): Servicio de Google Sheets
        spreadsheet_id (str): ID de la hoja de cálculo
        date_utils (DateUtils): Utilidades de fecha
        
    Returns:
        Optional[str]: Respuesta formateada o None si no aplica
    """
    if not sheets_service or not spreadsheet_id:
        logger.warning("Servicio de Google Sheets no disponible o ID de hoja no configurado.")
        return None # Devuelve None para que el RAG normal pueda continuar

    # Determinar el sheet_name a consultar dinámicamente
    from utils.date_utils import DateUtils
    date_utils = DateUtils()
    
    # Detectar si se menciona un mes específico en la consulta
    months_es = {
        'enero': 'ENERO', 'febrero': 'FEBRERO', 'marzo': 'MARZO', 'abril': 'ABRIL',
        'mayo': 'MAYO', 'junio': 'JUNIO', 'julio': 'JULIO', 'agosto': 'AGOSTO',
        'septiembre': 'SEPTIEMBRE', 'octubre': 'OCTUBRE', 'noviembre': 'NOVIEMBRE', 'diciembre': 'DICIEMBRE'
    }
    
    query_lower = query.lower()
    sheet_name_to_query = None
    
    # Buscar meses mencionados en la consulta
    for month_name, month_upper in months_es.items():
        if month_name in query_lower:
            sheet_name_to_query = month_upper
            logger.info(f"Mes detectado en consulta: {sheet_name_to_query}")
            break
    
    # Si no se detectó un mes específico, verificar referencias relativas
    if not sheet_name_to_query:
        if any(phrase in query_lower for phrase in ["mes que viene", "próximo mes", "proximo mes", "siguiente mes"]):
            # Obtener el mes siguiente
            current_month = date_utils.get_today().month
            next_month = current_month + 1 if current_month < 12 else 1
            next_month_name = list(months_es.values())[next_month - 1]
            sheet_name_to_query = next_month_name
            logger.info(f"Mes siguiente detectado: {sheet_name_to_query}")
        elif any(phrase in query_lower for phrase in ["mes pasado", "mes anterior", "último mes", "ultimo mes"]):
            # Obtener el mes anterior
            current_month = date_utils.get_today().month
            prev_month = current_month - 1 if current_month > 1 else 12
            prev_month_name = list(months_es.values())[prev_month - 1]
            sheet_name_to_query = prev_month_name
            logger.info(f"Mes anterior detectado: {sheet_name_to_query}")
        else:
            # Usar el mes actual
            sheet_name_to_query = date_utils.get_current_month_name()
            logger.info(f"Usando mes actual: {sheet_name_to_query}")
    else:
        logger.info(f"Consultando hoja del mes especificado: {sheet_name_to_query}")
    
    # Obtener todos los cursos de la hoja primero
    range_to_query = f"'{sheet_name_to_query}'!{SHEET_QUERY_CONFIG['range']}"
    logger.info(f"Consultando Google Sheet para cursos: ID {spreadsheet_id}, Rango: {range_to_query}")
    sheet_values = sheets_service.get_sheet_values(spreadsheet_id, range_to_query)

    if sheet_values is None:
        logger.error("Error al obtener datos de Google Sheets para cursos.")
        return None 
    
    if not sheet_values:
        logger.info(f"No se encontraron datos en la hoja '{sheet_name_to_query}' para el rango {SHEET_QUERY_CONFIG['range']}.")
        return None 

    # Parsear los datos a un formato más manejable
    parsed_courses = parse_sheet_course_data(sheet_values)
    if not parsed_courses:
        logger.info(f"No se pudieron parsear datos de cursos desde la hoja '{sheet_name_to_query}'.")
        return None

    query_lower = query.lower()
    
    # Verificar si la consulta es sobre cursos de la semana actual
    if any(phrase in query_lower for phrase in ["cursos esta semana", "cursos de esta semana", "hay cursos esta semana", 
                                              "cursos para esta semana", "qué hay esta semana"]):
        return handle_courses_this_week_query(parsed_courses, date_utils)
    
    # Verificar si la consulta es sobre cursos para un día específico
    weekday_queries = [
        ("lunes", "el lunes", "para el lunes", "del lunes"),
        ("martes", "el martes", "para el martes", "del martes"),
        ("miércoles", "el miércoles", "para el miércoles", "del miércoles"),
        ("jueves", "el jueves", "para el jueves", "del jueves"),
        ("viernes", "el viernes", "para el viernes", "del viernes"),
        ("sábado", "el sábado", "para el sábado", "del sábado"),
        ("domingo", "el domingo", "para el domingo", "del domingo"),
        ("hoy", "para hoy", "de hoy"),
        ("mañana", "para mañana", "de mañana")
    ]
    
    for weekday_terms in weekday_queries:
        base_term = weekday_terms[0]
        if any(term in query_lower for term in weekday_terms):
            # Si es "hoy", obtener el día de la semana actual
            if base_term == "hoy":
                base_term = date_utils.get_current_weekday_name()
            # Si es "mañana", obtener el día de mañana
            elif base_term == "mañana":
                tomorrow = date_utils.add_days(date_utils.get_today(), 1)
                base_term = date_utils.get_weekday_name(tomorrow)
            
            # Buscar también un curso específico en la consulta
            specific_course = None
            for keyword in ["curso", "actividad", "formulario", "cursos"]:
                if keyword in query_lower:
                    # Intentar extraer el nombre del curso después de la palabra clave
                    parts = query_lower.split(keyword, 1)
                    if len(parts) > 1 and parts[1].strip():
                        specific_course = parts[1].strip()
                        break
            
            return handle_weekday_course_query(parsed_courses, base_term, specific_course, date_utils)
    
    # Verificar si la consulta es sobre una fecha específica
    date = date_utils.parse_date_from_text(query)
    if date:
        date_str = f"{date.day:02d}/{date.month:02d}"
        return handle_date_course_query(parsed_courses, date_str, date, date_utils)
    
    # Búsqueda por nombre específico de curso
    # Nombres de cursos conocidos para búsqueda
    specific_course_keywords = {
        "primeros auxilios": "curso de primeros auxilios",
        "rcp": "curso de rcp",
        "suturas": "curso de suturas",
        "vacunación": "curso de vacunación"
    }

    target_course_name = None
    for keyword, full_name in specific_course_keywords.items():
        if keyword in query_lower:
            target_course_name = full_name
            break
    
    # Si la consulta pide un formulario y se identificó un curso
    if "formulario" in query_lower or "link para anotarme" in query_lower or "inscripción" in query_lower:
        if target_course_name:
            for course in parsed_courses:
                if target_course_name.lower() in course.get('nombre_actividad', '').lower():
                    if course.get('formulario', '').startswith('http'):
                        return f"📄 Aquí tienes el formulario para el {course.get('nombre_actividad', 'curso')}: {course.get('formulario')}"
                    else:
                        return f"ℹ️ Encontré información sobre el {course.get('nombre_actividad', 'curso')} pero no tiene un link de formulario directo en mis registros. Puedes consultar más detalles en la página."
            return f"🤔 No encontré un curso llamado '{target_course_name}' con formulario. ¿Podrías verificar el nombre?"
        else: # Formulario pero sin curso específico
             return "Si buscas el formulario para un curso específico, por favor dime el nombre del curso. 😊"

    # Si se identificó un curso específico en la consulta (y no es para formulario)
    if target_course_name:
        found_courses_info = []
        for course in parsed_courses:
            if target_course_name.lower() in course.get('nombre_actividad', '').lower():
                found_courses_info.append(format_course_info_for_response(course))
        if found_courses_info:
            return f"Esto es lo que encontré sobre el {target_course_name}:\n\n" + "\n\n".join(found_courses_info)
        else:
            return f"No encontré detalles para el curso '{target_course_name}' en la hoja. Verifica el nombre o consulta la página oficial."

    # Si es una consulta general sobre cursos (sin especificar uno)
    if any(keyword in query_lower for keyword in SHEET_COURSE_KEYWORDS if keyword not in specific_course_keywords):
        # Mostrar los cursos más próximos primero
        sorted_courses = sort_courses_by_proximity(parsed_courses, date_utils)
        response_intro = f"{random.choice(information_emojis)} Sobre los cursos en la hoja '{sheet_name_to_query}', esto es lo que tengo próximamente:\n"
        
        found_courses_info = []
        for course in sorted_courses[:3]: # Mostrar hasta 3 cursos
            found_courses_info.append(format_course_info_for_response(course))
        
        if found_courses_info:
            return response_intro + "\n\n".join(found_courses_info) + "\n\nPuedes preguntarme por un curso específico o su formulario si lo necesitas. 😉"

    return None # Si no coincide con nada específico de cursos


def handle_courses_this_week_query(parsed_courses: List[Dict[str, Any]], date_utils: DateUtils) -> str:
    """
    Maneja consultas sobre los cursos de la semana actual.
    
    Args:
        parsed_courses (List[Dict[str, Any]]): Lista de cursos parseados
        date_utils (DateUtils): Utilidades de fecha
        
    Returns:
        str: Respuesta formateada
    """
    this_week_courses = date_utils.get_courses_for_this_week(parsed_courses)
    
    if not this_week_courses:
        return "🗓️ No encontré cursos programados para esta semana en los registros."
    
    # Ordenar por fecha para mostrar los más próximos primero
    sorted_courses = sort_courses_by_proximity(this_week_courses, date_utils)
    
    response_parts = ["🗓️ Los cursos programados para esta semana son:"]
    for course in sorted_courses:
        response_parts.append(format_course_info_for_response(course))
    
    return "\n\n".join(response_parts)


def handle_weekday_course_query(parsed_courses: List[Dict[str, Any]], weekday_name: str, 
                               specific_course: str = None, date_utils: DateUtils = None) -> str:
    """
    Maneja consultas sobre cursos para un día específico de la semana.
    
    Args:
        parsed_courses (List[Dict[str, Any]]): Lista de cursos parseados
        weekday_name (str): Nombre del día de la semana
        specific_course (str): Nombre de curso específico (opcional)
        date_utils (DateUtils): Utilidades de fecha
        
    Returns:
        str: Respuesta formateada
    """
    # Limpiar la consulta para manejar casos como "formulario del curso del jueves"
    is_form_query = False
    clean_weekday_name = weekday_name.lower()
    
    # Verificar si es consulta de formulario
    if any(term in clean_weekday_name for term in ["formulario", "link", "enlace"]):
        is_form_query = True
    
    # Extraer solo el día de la semana
    day_only = None
    for day in ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo", "hoy", "mañana"]:
        if day in clean_weekday_name:
            day_only = day
            break
    
    if day_only:
        clean_weekday_name = day_only
    
    # Si se menciona 'del jueves', 'para hoy', etc. - limpiar para quedarnos solo con el día
    clean_weekday_name = clean_weekday_name.replace("del ", "").replace("para ", "").replace("este ", "").replace("esta ", "").strip()
    
    # Obtener los cursos para ese día
    weekday_courses = []
    if date_utils:
        weekday_courses = date_utils.get_courses_for_weekday(parsed_courses, clean_weekday_name)
    else:
        # Fallback si no tenemos date_utils
        weekday_abbr = {
            "lunes": "lun", "martes": "mar", "miércoles": "mié", "jueves": "jue", "viernes": "vie",
            "sábado": "sáb", "domingo": "dom"
        }.get(clean_weekday_name, "")
        
        for course in parsed_courses:
            # Buscar coincidencias directas o por abreviatura en el campo de fecha
            if (clean_weekday_name in course.get('fecha', '').lower() or 
                weekday_abbr in course.get('fecha', '').lower()):
                weekday_courses.append(course)
    
    if not weekday_courses:
        if specific_course:
            return f"🗓️ No encontré cursos de '{specific_course}' para {clean_weekday_name} en los próximos registros."
        else:
            return f"🗓️ No encontré cursos programados para {clean_weekday_name} en los próximos registros."
    
    # Si se solicitó un curso específico, filtrar la lista
    if specific_course:
        filtered_courses = []
        for course in weekday_courses:
            if specific_course.lower() in course.get('nombre_actividad', '').lower():
                filtered_courses.append(course)
        
        if filtered_courses:
            if len(filtered_courses) == 1 and 'formulario' in filtered_courses[0] and filtered_courses[0]['formulario'].startswith('http'):
                # Si solo hay un resultado y tiene formulario, mostrar el link directamente
                course = filtered_courses[0]
                return f"📄 Encontré el curso {course.get('nombre_actividad', '')} para {clean_weekday_name}. Aquí está el formulario: {course.get('formulario')}"
            else:
                # Si hay varios o no tiene formulario web, mostrar la información normal
                response_parts = [f"🗓️ Cursos de '{specific_course}' para {clean_weekday_name}:"]
                for course in filtered_courses:
                    response_parts.append(format_course_info_for_response(course))
                return "\n\n".join(response_parts)
        else:
            return f"🗓️ No encontré cursos que coincidan con '{specific_course}' para {clean_weekday_name}."
    
    # Si es una consulta específica de formulario sin especificar curso
    if is_form_query:
        # Buscar cursos con formulario
        courses_with_form = []
        for course in weekday_courses:
            if course.get('formulario', '').startswith('http'):
                courses_with_form.append(course)
        
        if courses_with_form:
            if len(courses_with_form) == 1:
                # Si solo hay un curso con formulario, mostrar directamente
                course = courses_with_form[0]
                return f"📄 Encontré el formulario para el curso {course.get('nombre_actividad', '')} del {clean_weekday_name}: {course.get('formulario')}"
            else:
                # Si hay varios, mostrar todos los disponibles
                response_parts = [f"📄 Formularios disponibles para {clean_weekday_name}:"]
                for course in courses_with_form:
                    response_parts.append(f"- {course.get('nombre_actividad', '')}: {course.get('formulario')}")
                return "\n\n".join(response_parts)
        else:
            return f"🔍 No encontré formularios para los cursos del {clean_weekday_name}. ¿Te interesa un curso específico?"
    
    # Mostrar todos los cursos para ese día
    response_parts = [f"🗓️ Cursos programados para {clean_weekday_name}:"]
    for course in weekday_courses:
        response_parts.append(format_course_info_for_response(course))
    
    return "\n\n".join(response_parts)


def handle_date_course_query(parsed_courses: List[Dict[str, Any]], date_str: str, 
                            date_obj: datetime.date, date_utils: DateUtils) -> str:
    """
    Maneja consultas sobre cursos para una fecha específica.
    
    Args:
        parsed_courses (List[Dict[str, Any]]): Lista de cursos parseados
        date_str (str): Fecha en formato dd/mm
        date_obj (datetime.date): Objeto de fecha
        date_utils (DateUtils): Utilidades de fecha
        
    Returns:
        str: Respuesta formateada
    """
    # Buscar cursos que coincidan con la fecha exacta
    date_courses = []
    for course in parsed_courses:
        if course.get('fecha') == date_str:
            date_courses.append(course)
    
    if not date_courses:
        weekday_name = date_utils.get_weekday_name(date_obj)
        formatted_date = date_utils.get_formatted_date(date_obj)
        return f"🗓️ No encontré cursos programados para el {formatted_date} ({weekday_name})."
    
    formatted_date = date_utils.get_formatted_date(date_obj)
    response_parts = [f"🗓️ Cursos programados para el {formatted_date}:"]
    
    for course in date_courses:
        response_parts.append(format_course_info_for_response(course))
    
    return "\n\n".join(response_parts) 