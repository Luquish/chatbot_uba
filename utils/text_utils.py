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


