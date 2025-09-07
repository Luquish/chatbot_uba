import re
from unidecode import unidecode
from typing import Optional


def normalize_text(text: str) -> str:
    """
    Normaliza texto para matching robusto:
    - Minúsculas
    - Remueve tildes
    - Limpia símbolos no alfanuméricos (mantiene @._- para emails)
    - Colapsa espacios
    """
    s = unidecode(str(text or "")).lower().strip()
    s = re.sub(r"[^a-z0-9\s@._-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_day_token(text: str) -> Optional[str]:
    """
    Detecta día de la semana en español y devuelve un token de columna: L/M/W/J/V.
    Incluye abreviaturas y variantes con/ sin tilde.
    """
    t = normalize_text(text)
    day_map = {
        'lunes': 'L', 'lun': 'L',
        'martes': 'M', 'mar': 'M',
        'miercoles': 'W', 'mier': 'W', 'mie': 'W',
        'jueves': 'J', 'jue': 'J',
        'viernes': 'V', 'vie': 'V'
    }
    for k, v in day_map.items():
        if k in t:
            return v
    if 'hoy' in t:
        import datetime
        weekday = datetime.datetime.now().weekday()
        return ['L', 'M', 'W', 'J', 'V', None, None][weekday]
    return None


def extract_catedra_number(text: str) -> Optional[str]:
    """
    Extrae el número de cátedra desde un texto si aparece como "catedra N" o "cátedra N".
    Usa texto normalizado para robustez.
    """
    t = normalize_text(text)
    m = re.search(r"catedra\s*(\d+)", t)
    if m:
        return m.group(1)
    return None


def is_count_catedras_query(text: str) -> bool:
    """
    Detecta consultas del tipo "¿cuántas cátedras ...?" en texto normalizado.
    """
    t = normalize_text(text)
    return ("cuantas" in t) and ("catedra" in t or "catedras" in t)


def strip_markdown_emphasis(text: str) -> str:
    """
    Elimina énfasis Markdown común (** **, * *, __ __, _ _).
    """
    if not text:
        return text
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"\_\_(.+?)\_\_", r"\1", text)
    text = re.sub(r"\_(.+?)\_", r"\1", text)
    return text


def preprocess_for_embedding(text: str) -> str:
    """
    Preprocesa texto específicamente para generar embeddings de alta calidad.
    
    Args:
        text: Texto a preprocesar
        
    Returns:
        Texto preprocesado optimizado para embeddings
    """
    if not text:
        return ""
    
    # 1. Limpiar caracteres especiales y formato PDF
    text = re.sub(r'\[Documento:.*?\]', '', text)  # Remover metadatos de documento
    text = re.sub(r'<sup>\d+</sup>', '', text)  # Remover referencias como <sup>1</sup>
    text = re.sub(r'##\s*', '', text)  # Remover headers markdown
    text = re.sub(r'\n+', ' ', text)  # Reemplazar saltos de línea con espacios
    
    # 2. Normalizar espacios y caracteres
    text = re.sub(r'\s+', ' ', text)  # Colapsar espacios múltiples
    text = text.strip()
    
    # 3. Limpiar caracteres de control y no imprimibles
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 4. Preservar estructura semántica importante
    # Mantener números, fechas, artículos legales
    text = re.sub(r'Art\.?\s*(\d+)', r'Artículo \1', text)  # Normalizar referencias a artículos
    text = re.sub(r'Resolución\s*\(CS\)\s*(\d+)', r'Resolución CS \1', text)  # Normalizar resoluciones
    
    # 5. Limpiar pero preservar contenido académico
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/]', ' ', text)  # Mantener puntuación importante
    text = re.sub(r'\s+', ' ', text)  # Limpiar espacios nuevamente
    
    return text.strip()


def clean_query_for_embedding(query: str) -> str:
    """
    Limpia consultas de usuario para mejorar matching con embeddings.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Consulta limpia optimizada para embeddings
    """
    if not query:
        return ""
    
    # Normalizar consulta
    query = query.strip()
    
    # Remover signos de interrogación al inicio/final
    query = re.sub(r'^[¿\?]+|[¿\?]+$', '', query)
    
    # Normalizar variaciones comunes
    query = re.sub(r'\bcomo\b', 'cómo', query, flags=re.IGNORECASE)
    query = re.sub(r'\bdonde\b', 'dónde', query, flags=re.IGNORECASE)
    query = re.sub(r'\bque\b', 'qué', query, flags=re.IGNORECASE)
    query = re.sub(r'\bcuando\b', 'cuándo', query, flags=re.IGNORECASE)
    
    # Limpiar espacios
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query


