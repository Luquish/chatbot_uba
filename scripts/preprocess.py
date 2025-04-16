import os
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from io import StringIO
import re
from pdfminer.high_level import extract_text
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        """
        Inicializa el preprocesador de documentos.
        
        Args:
            raw_dir (str): Directorio con documentos sin procesar
            processed_dir (str): Directorio para guardar documentos procesados
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extrae texto de un archivo PDF usando pdfminer de manera más robusta.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            str: Texto extraído del PDF
        """
        try:
            logger.info(f"Extrayendo texto de: {pdf_path}")
            
            # Método 1: Usar extract_text (más simple pero a veces menos preciso)
            text = extract_text(pdf_path)
            
            # Si el texto está vacío o es muy corto, usar método alternativo
            if not text or len(text) < 100:
                logger.warning(f"Primera extracción produjo texto insuficiente, usando método alternativo para {pdf_path}")
                
                # Método 2: Más detallado usando componentes de pdfminer
                output_string = StringIO()
                with open(pdf_path, 'rb') as in_file:
                    parser = PDFParser(in_file)
                    doc = PDFDocument(parser)
                    rsrcmgr = PDFResourceManager()
                    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                    interpreter = PDFPageInterpreter(rsrcmgr, device)
                    for page in PDFPage.create_pages(doc):
                        interpreter.process_page(page)
                
                text = output_string.getvalue()
            
            # Verificación básica
            if not text:
                logger.error(f"No se pudo extraer texto de {pdf_path}")
                return ""
                
            logger.info(f"Texto extraído exitosamente: {len(text)} caracteres")
            return text
            
        except Exception as e:
            logger.error(f"Error al extraer texto de {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído de manera más exhaustiva.
        
        Args:
            text (str): Texto a limpiar
            
        Returns:
            str: Texto limpio
        """
        if not text:
            return ""
            
        # Eliminar caracteres de control excepto saltos de línea y tabulaciones
        text = ''.join(char for char in text if char == '\n' or char == '\t' or char.isprintable())
        
        # Reemplazar múltiples espacios en blanco por uno solo
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Reemplazar múltiples saltos de línea por uno solo
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Eliminar espacios al inicio y final de cada línea
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Eliminar líneas que solo contengan números o sean muy cortas (menos de 3 caracteres)
        lines = [line for line in text.split('\n') if len(line) > 3 and not line.strip().isdigit()]
        text = '\n'.join(lines)
        
        return text.strip()

    def convert_to_markdown(self, text: str) -> str:
        """
        Convierte el texto extraído del PDF a formato Markdown mejorado para resoluciones universitarias.
        
        Args:
            text (str): Texto extraído del PDF
            
        Returns:
            str: Texto en formato Markdown enriquecido
        """
        if not text:
            return ""
        
        # Patrones para identificar estructura específica de resoluciones universitarias
        title_pattern = re.compile(r'^([A-Z][A-Z\s]{5,})(?:\n|$)')
        resolution_pattern = re.compile(r'Resolución\s+(?:\([A-Z]+\))?\s*(\d+\/\d+)')
        article_pattern = re.compile(r'^Art(?:ículo|\.)\s*(\d+\º?)\.?', re.IGNORECASE)
        section_pattern = re.compile(r'^(?:CAPÍTULO|TÍTULO|SECCIÓN)\s+([IVX\d]+)\.?', re.IGNORECASE)
        subsection_pattern = re.compile(r'^(?:[a-z]\)|[0-9]+\.|[A-Z]\))', re.IGNORECASE)
        visto_pattern = re.compile(r'^\s*Visto', re.IGNORECASE)
        considerando_pattern = re.compile(r'^\s*Considerando:', re.IGNORECASE)
        resuelve_pattern = re.compile(r'^\s*(?:El\s+Consejo\s+Superior.*?\n\s*)?Resuelve:', re.IGNORECASE)
        anexo_pattern = re.compile(r'^\s*Anexo\s*$', re.IGNORECASE)
        inciso_pattern = re.compile(r'^\s*([a-z]\))', re.IGNORECASE)
        
        # Separar el texto en líneas para procesamiento
        lines = text.split('\n')
        markdown_lines = []
        in_list = False
        title_found = False
        resolution_found = False
        visto_found = False
        considerando_found = False
        resuelve_found = False
        anexo_found = False
        rector_line = False
        
        # Procesar línea por línea
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Saltar líneas vacías al inicio
            if not line and not title_found:
                i += 1
                continue
            
            # Detectar título principal
            if not title_found and title_pattern.match(line):
                markdown_lines.append(f"# {line}")
                title_found = True
                i += 1
                continue
            
            # Detectar número de resolución
            if title_found and not resolution_found and resolution_pattern.search(line):
                markdown_lines.append(f"**{line}**")
                resolution_found = True
                i += 1
                continue
            
            # Detectar sección "Visto"
            if resolution_found and not visto_found and visto_pattern.match(line):
                markdown_lines.append("\n_Visto_:")
                
                # Capturar todo el texto de "Visto" hasta "Considerando" o "Resuelve"
                visto_text = []
                j = i
                while j < len(lines) and not considerando_pattern.match(lines[j]) and not resuelve_pattern.match(lines[j]):
                    if lines[j].strip() and not visto_pattern.match(lines[j]):
                        visto_text.append(lines[j].strip())
                    j += 1
                
                # Formatear el texto "Visto" como lista si es apropiado
                if any(re.match(r'^-|\d+\)|\([a-z]\)', line) for line in visto_text):
                    for vt in visto_text:
                        if re.match(r'^-', vt):
                            markdown_lines.append(f"- {vt[1:].strip()}")
                        elif re.match(r'^\d+\)|\([a-z]\)', vt):
                            markdown_lines.append(f"- {vt}")
                        else:
                            markdown_lines.append(f"- {vt}")
                else:
                    markdown_lines.append(" ".join(visto_text))
                
                markdown_lines.append("\n---\n")
                visto_found = True
                i = j
                continue
            
            # Detectar sección "Considerando"
            if (resolution_found or visto_found) and not considerando_found and considerando_pattern.match(line):
                markdown_lines.append("**Considerando:**\n")
                
                # Capturar todo el considerando hasta "Resuelve"
                considerando_text = []
                j = i + 1
                while j < len(lines) and not resuelve_pattern.match(lines[j]):
                    if lines[j].strip() and not "El Consejo Superior" in lines[j]:
                        # Verificar si es un "Que" inicial para formato de lista
                        if lines[j].strip().startswith("Que") or lines[j].strip().startswith("que"):
                            considerando_text.append(f"- {lines[j].strip()}")
                        else:
                            considerando_text.append(lines[j].strip())
                    j += 1
                
                # Formatear el considerando como texto normal o lista según corresponda
                for ct in considerando_text:
                    markdown_lines.append(ct)
                
                considerando_found = True
                i = j - 1  # Retroceder uno para que el próximo ciclo capture "Resuelve"
                i += 1
                continue
            
            # Detectar línea "El Consejo Superior" antes de Resuelve
            if not resuelve_found and "El Consejo Superior" in line:
                markdown_lines.append(f"\n**{line}**\n")
                i += 1
                continue
            
            # Detectar sección "Resuelve"
            if (visto_found or considerando_found or resolution_found) and not resuelve_found and resuelve_pattern.match(line):
                markdown_lines.append("\n---\n")
                markdown_lines.append("## Resuelve:\n")
                resuelve_found = True
                i += 1
                continue
            
            # Detectar sección "Anexo"
            if not anexo_found and anexo_pattern.match(line):
                markdown_lines.append("\n---\n")
                markdown_lines.append("## Anexo\n")
                anexo_found = True
                i += 1
                continue
            
            # Detectar artículos
            article_match = article_pattern.match(line)
            if article_match:
                # Terminar lista anterior si existía
                if in_list:
                    in_list = False
                    markdown_lines.append("")
                
                # Agregar artículo con formato
                markdown_lines.append(f"### {line}")
                
                # Capturar todo el contenido del artículo hasta el próximo artículo
                j = i + 1
                article_content = []
                while j < len(lines) and not article_pattern.match(lines[j].strip()) and not anexo_pattern.match(lines[j].strip()):
                    if lines[j].strip():
                        article_content.append(lines[j].strip())
                    j += 1
                
                # Procesar el contenido del artículo
                if article_content:
                    # Verificar si tiene incisos
                    has_incisos = any(inciso_pattern.match(line) for line in article_content)
                    
                    if has_incisos:
                        # Formatear incisos como lista con viñetas
                        current_text = ""
                        for ac in article_content:
                            inciso_match = inciso_pattern.match(ac)
                            if inciso_match:
                                # Si hay texto previo, añadirlo
                                if current_text:
                                    markdown_lines.append(current_text)
                                    current_text = ""
                                
                                # Añadir el inciso formateado
                                markdown_lines.append(f"- **{inciso_match.group(1)}** {ac[inciso_match.end():].strip()}")
                            else:
                                # Acumular texto normal o añadirlo a la última viñeta
                                if markdown_lines and markdown_lines[-1].startswith("- **"):
                                    markdown_lines[-1] += f" {ac}"
                                else:
                                    current_text += f" {ac}"
                        
                        # Añadir el texto final si quedó algo
                        if current_text.strip():
                            markdown_lines.append(current_text.strip())
                    else:
                        # Añadir como texto normal
                        markdown_lines.append(" ".join(article_content))
                
                markdown_lines.append("")  # Espacio después del artículo
                i = j
                continue
            
            # Detectar firma del rector
            if "Rector" in line and "Shuberoff" in line:
                markdown_lines.append(f"\n**{line}**")
                rector_line = True
                i += 1
                continue
            
            # Procesar líneas normales
            if line:
                # Si es una línea vacía, mantener el formato
                if not in_list:
                    markdown_lines.append(line)
                else:
                    # Estamos en una lista, intentar mantener formato de lista
                    if subsection_pattern.match(line):
                        markdown_lines.append(f"- {line}")
                    else:
                        markdown_lines.append(line)
                        in_list = False
            else:
                # Mantener líneas en blanco para preservar formato
                markdown_lines.append("")
            
            i += 1
        
        # Eliminar líneas en blanco repetidas
        clean_markdown = []
        prev_empty = False
        for line in markdown_lines:
            if not line.strip():
                if not prev_empty:
                    clean_markdown.append("")
                    prev_empty = True
            else:
                clean_markdown.append(line)
                prev_empty = False
        
        return "\n".join(clean_markdown)

    def split_into_chunks(self, text: str, filename: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        if not text:
            return []
        
        # Patrones para identificar estructura del documento
        article_pattern = re.compile(r'(?i)(?:^|\n)(?:Art(?:ículo|\.)\s*\d+\º?\.?)', re.MULTILINE)
        section_pattern = re.compile(r'(?i)(?:^|\n)(?:CAPÍTULO|TÍTULO|SECCIÓN)\s+[IVX\d]+\.?', re.MULTILINE)
        subsection_pattern = re.compile(r'(?i)(?:^|\n)(?:[a-z]\)|[0-9]+\.|[A-Z]\))', re.MULTILINE)
        
        # Primero dividir por artículos
        chunks = []
        article_matches = list(article_pattern.finditer(text))
        
        if not article_matches:
            # Si no hay artículos, tratar todo el texto como un chunk
            if len(text.split()) <= chunk_size:
                doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                chunks.append(doc_prefix + text)
            else:
                # Dividir por párrafos
                paragraphs = text.split('\n\n')
                current_chunk = []
                current_size = 0
                
                for paragraph in paragraphs:
                    words = paragraph.split()
                    if current_size + len(words) <= chunk_size:
                        current_chunk.append(paragraph)
                        current_size += len(words)
                    else:
                        if current_chunk:
                            doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                            chunks.append(doc_prefix + '\n\n'.join(current_chunk))
                        current_chunk = [paragraph]
                        current_size = len(words)
                
                if current_chunk:
                    doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                    chunks.append(doc_prefix + '\n\n'.join(current_chunk))
        else:
            # Procesar cada artículo
            for i, match in enumerate(article_matches):
                start = match.start()
                end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
                article_text = text[start:end].strip()
                
                # Dividir el artículo en subsecciones si es necesario
                subsections = []
                current_subsection = []
                
                for line in article_text.split('\n'):
                    if subsection_pattern.match(line) and current_subsection:
                        subsections.append('\n'.join(current_subsection))
                        current_subsection = []
                    current_subsection.append(line)
                
                if current_subsection:
                    subsections.append('\n'.join(current_subsection))
                
                if not subsections:
                    subsections = [article_text]
                
                # Procesar cada subsección
                for subsection in subsections:
                    words = subsection.split()
                    if len(words) <= chunk_size:
                        doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                        chunks.append(doc_prefix + subsection)
                    else:
                        # Dividir subsección grande en chunks más pequeños
                        current_chunk = []
                        current_size = 0
                        
                        for word in words:
                            current_chunk.append(word)
                            current_size += 1
                            
                            if current_size >= chunk_size:
                                # Buscar un punto final cercano
                                look_back = min(10, len(current_chunk))
                                cut_point = -1
                                
                                for j in range(look_back):
                                    if current_chunk[-j-1].endswith('.'):
                                        cut_point = len(current_chunk) - j - 1
                                        break
                                
                                if cut_point == -1:
                                    cut_point = len(current_chunk)
                                
                                doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                                chunks.append(doc_prefix + ' '.join(current_chunk[:cut_point]))
                                
                                # Mantener palabras para contexto
                                current_chunk = current_chunk[max(0, cut_point-overlap):]
                                current_size = len(current_chunk)
                        
                        if current_chunk:
                            doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                            chunks.append(doc_prefix + ' '.join(current_chunk))
        
        # Post-procesamiento para evitar chunks muy pequeños
        processed_chunks = []
        min_chunk_size = 50  # palabras
        
        for i, chunk in enumerate(chunks):
            chunk_words = chunk.split()
            if len(chunk_words) < min_chunk_size and i > 0:
                # Combinar chunks pequeños con el anterior
                previous_chunk = processed_chunks[-1]
                combined_chunk = previous_chunk + " " + chunk
                processed_chunks[-1] = combined_chunk
            else:
                processed_chunks.append(chunk)
        
        logger.info(f"Documento dividido en {len(processed_chunks)} chunks")
        return processed_chunks

    def process_document(self, pdf_path: Path) -> Dict:
        """
        Procesa un documento PDF y lo convierte a Markdown.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            Dict: Diccionario con metadatos y chunks del documento
        """
        logger.info(f"Procesando documento: {pdf_path}")
        
        # Extraer texto del PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"No se pudo extraer texto de {pdf_path}")
            return None
        
        # Limpiar texto
        clean_text = self.clean_text(text)
        
        # Convertir a Markdown
        markdown_text = self.convert_to_markdown(clean_text)
        
        # Guardar archivo Markdown
        markdown_path = self.processed_dir / f"{pdf_path.stem}.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        logger.info(f"Archivo Markdown guardado en: {markdown_path}")
        
        # Generar chunks basados en la estructura Markdown
        chunks = self.split_markdown_into_chunks(markdown_text, pdf_path.name)
        
        word_count = len(clean_text.split())
        logger.info(f"Documento procesado: {word_count} palabras, {len(chunks)} chunks")
        
        return {
            'filename': pdf_path.name,
            'markdown_file': str(markdown_path),
            'num_chunks': len(chunks),
            'total_words': word_count,
            'chunks': chunks
        }

    def split_markdown_into_chunks(self, markdown_text: str, filename: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        """
        Divide el texto Markdown en chunks respetando la estructura del documento.
        
        Args:
            markdown_text (str): Texto en formato Markdown
            filename (str): Nombre del archivo original
            chunk_size (int): Tamaño objetivo de cada chunk
            overlap (int): Cantidad de palabras de solapamiento
            
        Returns:
            List[str]: Lista de chunks
        """
        if not markdown_text:
            return []
        
        chunks = []
        lines = markdown_text.split('\n')
        
        # Detectar patrones markdown específicos
        title_pattern = re.compile(r'^#\s+')  # Título principal
        section_pattern = re.compile(r'^##\s+')  # Secciones
        article_pattern = re.compile(r'^###\s+Art(?:ículo|\.)\s*(\d+\º?)\.?', re.IGNORECASE)  # Artículos
        
        # Primero identificar las secciones principales y los artículos
        section_indices = []
        article_indices = []
        
        for i, line in enumerate(lines):
            if title_pattern.match(line):
                section_indices.append(i)
            elif section_pattern.match(line):
                section_indices.append(i)
            elif article_pattern.match(line):
                article_indices.append(i)
        
        # Si hay artículos, usar eso como división principal
        if article_indices:
            # Agregar el índice final para facilitar los slices
            article_indices.append(len(lines))
            
            for i in range(len(article_indices) - 1):
                start_idx = article_indices[i]
                end_idx = article_indices[i + 1]
                
                # Extraer artículo completo
                article_lines = lines[start_idx:end_idx]
                article_text = '\n'.join(article_lines)
                
                # Si el artículo es pequeño, dejarlo como un chunk
                if len(article_text.split()) <= chunk_size:
                    doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                    chunks.append(doc_prefix + article_text)
                else:
                    # Dividir artículos grandes en sub-chunks
                    current_chunk = []
                    current_size = 0
                    
                    for line in article_lines:
                        line_words = len(line.split())
                        
                        if current_size + line_words <= chunk_size:
                            current_chunk.append(line)
                            current_size += line_words
                        else:
                            # Guardar el chunk actual si tiene contenido
                            if current_chunk:
                                doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                                chunks.append(doc_prefix + '\n'.join(current_chunk))
                            
                            # Iniciar un nuevo chunk manteniendo el nombre del artículo para contexto
                            current_chunk = [article_lines[0]]  # Agregar siempre la línea del artículo
                            if line != article_lines[0]:  # Evitar duplicar el nombre del artículo
                                current_chunk.append(line)
                                current_size = line_words + len(article_lines[0].split())
                            else:
                                current_size = line_words
                    
                    # Guardar el último chunk del artículo
                    if current_chunk:
                        doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                        chunks.append(doc_prefix + '\n'.join(current_chunk))
        
        # Si no hay artículos, dividir por secciones
        elif section_indices:
            section_indices.append(len(lines))
            
            for i in range(len(section_indices) - 1):
                start_idx = section_indices[i]
                end_idx = section_indices[i + 1]
                
                section_lines = lines[start_idx:end_idx]
                section_text = '\n'.join(section_lines)
                
                # Si la sección es pequeña, usarla como chunk
                if len(section_text.split()) <= chunk_size:
                    doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                    chunks.append(doc_prefix + section_text)
                else:
                    # Dividir secciones grandes
                    current_chunk = [section_lines[0]]  # Mantener el encabezado
                    current_size = len(section_lines[0].split())
                    
                    for line in section_lines[1:]:
                        line_words = len(line.split())
                        
                        if current_size + line_words <= chunk_size:
                            current_chunk.append(line)
                            current_size += line_words
                        else:
                            # Guardar chunk actual
                            doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                            chunks.append(doc_prefix + '\n'.join(current_chunk))
                            
                            # Iniciar nuevo chunk con contexto
                            current_chunk = [section_lines[0], line]
                            current_size = len(section_lines[0].split()) + line_words
                    
                    # Guardar el último chunk de la sección
                    if current_chunk:
                        doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                        chunks.append(doc_prefix + '\n'.join(current_chunk))
        
        # Si no hay artículos ni secciones, usar el método normal
        else:
            current_section = []
            current_size = 0
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Si es un título o encabezado, siempre iniciar nuevo chunk
                if title_pattern.match(line) or section_pattern.match(line):
                    if current_section:
                        doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                        chunks.append(doc_prefix + '\n'.join(current_section))
                    current_section = [line]
                    current_size = len(line.split())
                    i += 1
                    continue
                
                # Procesar líneas normales
                words = line.split() if line else []
                if current_size + len(words) <= chunk_size:
                    if line or current_section:  # No añadir líneas vacías al inicio
                        current_section.append(line)
                    current_size += len(words)
                    i += 1
                else:
                    # Si el chunk actual está vacío, forzar la inclusión de esta línea
                    if not current_section:
                        current_section.append(line)
                        i += 1
                    
                    # Guardar chunk actual
                    if current_section:
                        doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                        chunks.append(doc_prefix + '\n'.join(current_section))
                    
                    # Mantener contexto para el siguiente chunk (encabezados importantes)
                    current_section = []
                    for j in range(max(0, i - overlap), i):
                        if title_pattern.match(lines[j]) or section_pattern.match(lines[j]):
                            current_section.append(lines[j])
                    current_size = sum(len(line.split()) for line in current_section)
            
            # Añadir el último chunk si existe
            if current_section:
                doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                chunks.append(doc_prefix + '\n'.join(current_section))
        
        # Post-procesamiento para evitar chunks muy pequeños
        processed_chunks = []
        min_chunk_size = 50  # palabras
        
        for i, chunk in enumerate(chunks):
            chunk_words = len(chunk.split())
            
            if chunk_words < min_chunk_size and i > 0 and not title_pattern.search(chunk) and not section_pattern.search(chunk):
                # Combinar chunks pequeños con el anterior
                previous_chunk = processed_chunks[-1]
                combined_chunk = previous_chunk + "\n\n" + chunk
                processed_chunks[-1] = combined_chunk
            else:
                processed_chunks.append(chunk)
        
        logger.info(f"Documento dividido en {len(processed_chunks)} chunks")
        return processed_chunks

    def process_all_documents(self):
        """Procesa todos los documentos PDF en el directorio raw con mejor manejo de errores."""
        pdf_files = list(self.raw_dir.glob('**/*.pdf'))
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF")
        
        if not pdf_files:
            logger.warning(f"No se encontraron archivos PDF en {self.raw_dir}")
            return
        
        processed_data = []
        for pdf_path in tqdm(pdf_files, desc="Procesando documentos"):
            try:
                logger.info(f"Procesando: {pdf_path}")
                result = self.process_document(pdf_path)
                if result:
                    processed_data.append(result)
                    logger.info(f"Documento procesado exitosamente: {pdf_path.name}")
                else:
                    logger.error(f"Error al procesar documento: {pdf_path}")
            except Exception as e:
                logger.error(f"Excepción al procesar {pdf_path}: {str(e)}", exc_info=True)
                
        # Guardar resultados en CSV
        if processed_data:
            df = pd.DataFrame(processed_data)
            output_path = self.processed_dir / 'processed_documents.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Datos procesados guardados en {output_path}")
            
            # Guardar chunks individuales
            chunks_dir = self.processed_dir / 'chunks'
            chunks_dir.mkdir(exist_ok=True)
            
            total_chunks = 0
            for doc in processed_data:
                for i, chunk in enumerate(doc['chunks']):
                    chunk_path = chunks_dir / f"{doc['filename']}_chunk_{i}.txt"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)
                    total_chunks += 1
            
            logger.info(f"Total de chunks guardados: {total_chunks}")
        else:
            logger.warning("No se procesó ningún documento correctamente")

def main():
    """Función principal para ejecutar el preprocesamiento."""
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    logger.info(f"Iniciando preprocesamiento. Directorio raw: {raw_dir}, Directorio procesado: {processed_dir}")
    
    preprocessor = DocumentPreprocessor(raw_dir, processed_dir)
    preprocessor.process_all_documents()

if __name__ == "__main__":
    main() 