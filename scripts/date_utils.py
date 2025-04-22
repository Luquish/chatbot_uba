import re
from datetime import datetime, timedelta
from typing import Tuple, Optional

class DateUtils:
    def __init__(self):
        self.meses = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }
        
    def month_to_number(self, month: str) -> str:
        """Convierte un nombre de mes en español a su número correspondiente (01-12)."""
        month = month.lower().strip()
        return self.meses.get(month, '')

    def extract_dates_from_query(self, query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Extrae fechas de inicio y fin de una consulta en español.
        Retorna una tupla de (fecha_inicio, fecha_fin).
        """
        query = query.lower()
        start_date = None
        end_date = None
        
        # Patrones de fecha en español
        fecha_patron = r'(\d{1,2})\s+(?:de\s+)?([a-zá-úñ]+)(?:\s+(?:de\s+)?(\d{4}|\d{2}))?'
        
        # Buscar fechas específicas en el texto
        fechas = re.finditer(fecha_patron, query)
        fechas_encontradas = []
        
        for match in fechas:
            dia, mes, año = match.groups()
            if not año:
                año = str(datetime.now().year)
            elif len(año) == 2:
                año = '20' + año
                
            mes_num = self.month_to_number(mes)
            if mes_num:
                try:
                    fecha = datetime.strptime(f"{dia.zfill(2)}/{mes_num}/{año}", "%d/%m/%Y")
                    fechas_encontradas.append(fecha)
                except ValueError:
                    continue

        # Si se encontraron fechas específicas
        if len(fechas_encontradas) >= 2:
            start_date = min(fechas_encontradas)
            end_date = max(fechas_encontradas)
        elif len(fechas_encontradas) == 1:
            if "hasta" in query:
                end_date = fechas_encontradas[0]
                start_date = datetime.now()
            else:
                start_date = fechas_encontradas[0]
                end_date = start_date + timedelta(days=1)

        # Palabras clave para rangos de tiempo
        if not (start_date and end_date):
            hoy = datetime.now()
            if any(palabra in query for palabra in ["esta semana", "semana actual"]):
                start_date = hoy - timedelta(days=hoy.weekday())
                end_date = start_date + timedelta(days=6)
            elif "próxima semana" in query:
                start_date = hoy + timedelta(days=7-hoy.weekday())
                end_date = start_date + timedelta(days=6)
            elif "mes actual" in query or "este mes" in query:
                start_date = hoy.replace(day=1)
                if hoy.month == 12:
                    end_date = hoy.replace(year=hoy.year+1, month=1, day=1) - timedelta(days=1)
                else:
                    end_date = hoy.replace(month=hoy.month+1, day=1) - timedelta(days=1)
            elif "próximo mes" in query:
                if hoy.month == 12:
                    start_date = hoy.replace(year=hoy.year+1, month=1, day=1)
                    end_date = start_date.replace(month=2, day=1) - timedelta(days=1)
                else:
                    start_date = hoy.replace(month=hoy.month+1, day=1)
                    if hoy.month == 11:
                        end_date = start_date.replace(year=hoy.year+1, month=1, day=1) - timedelta(days=1)
                    else:
                        end_date = start_date.replace(month=hoy.month+2, day=1) - timedelta(days=1)

        return start_date, end_date

    def format_date_for_api(self, date: datetime) -> str:
        """Formatea una fecha para la API de Google Calendar."""
        return date.isoformat() + 'Z'  # Formato ISO 8601 