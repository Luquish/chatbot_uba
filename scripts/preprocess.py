import os
import logging
import subprocess
from pathlib import Path
import re
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarkerPreprocessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        """
        Inicializa el preprocesador basado en Marker PDF.
        
        Args:
            raw_dir (str): Directorio con documentos sin procesar
            processed_dir (str): Directorio para guardar documentos procesados
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar que marker esté instalado
        try:
            result = subprocess.run(['marker_single', '--help'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True,
                                   check=False)
            
            if result.returncode != 0:
                logger.error(f"Error al verificar marker_single: {result.stderr}")
                raise RuntimeError("marker_single no está disponible o no funciona correctamente")
                
            logger.info("Marker detectado correctamente en el sistema")
        except FileNotFoundError:
            logger.error("No se encontró el comando marker_single")
            logger.error("Asegúrate de que marker-pdf esté instalado: pip install marker-pdf")
            raise RuntimeError("Marker no está instalado correctamente")
    
    def process_document(self, pdf_path: Path) -> Dict:
        """
        Procesa un documento PDF usando Marker y lo convierte a Markdown.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            Dict: Diccionario con metadatos y chunks del documento
        """
        logger.info(f"Procesando documento con Marker: {pdf_path}")
        
        # Crear un subdirectorio específico para este documento
        output_subdir = self.processed_dir / pdf_path.stem
        output_subdir.mkdir(exist_ok=True)
        
        # Preparar los argumentos para marker_single
        marker_cmd = [
            "marker_single",
            str(pdf_path),
            "--output_dir", str(output_subdir),
            "--output_format", "markdown",
            "--paginate_output",
            "--force_ocr"
        ]
        
        # Ejecutar marker_single para convertir el PDF a Markdown
        try:
            logger.info(f"Ejecutando: {' '.join(marker_cmd)}")
            result = subprocess.run(
                marker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Error al ejecutar marker_single: {result.stderr}")
                return None
                
            logger.info(f"Marker completado exitosamente para {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Excepción al ejecutar marker_single: {str(e)}")
            return None
        
        # Marker crea una subcarpeta con el mismo nombre del archivo que debe eliminarse
        nested_dir = output_subdir / pdf_path.stem
        
        # Variable para almacenar la ruta al archivo Markdown encontrado
        markdown_path = output_subdir / f"{pdf_path.stem}.md"
        
        # Verificar si existe la carpeta anidada y manejarla
        if nested_dir.exists() and nested_dir.is_dir():
            logger.info(f"Detectada carpeta anidada: {nested_dir}")
            
            # 1. Buscar y mover archivos Markdown
            markdown_files = list(nested_dir.glob("*.md"))
            if markdown_files:
                original_markdown_path = markdown_files[0]
                try:
                    # Leer el contenido del archivo original
                    with open(original_markdown_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Escribir el contenido en la ubicación correcta (nivel superior)
                    with open(markdown_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
                    logger.info(f"Archivo Markdown movido a: {markdown_path}")
                except Exception as e:
                    logger.error(f"Error al mover el archivo Markdown: {str(e)}")
                    return None
            else:
                logger.error(f"No se encontraron archivos Markdown en: {nested_dir}")
                return None
                
            # 2. Buscar y mover archivos de metadatos JSON
            json_files = list(nested_dir.glob(f"{pdf_path.stem}_meta.json"))
            if json_files:
                try:
                    original_json_path = json_files[0]
                    target_json_path = output_subdir / f"{pdf_path.stem}_meta.json"
                    
                    # Leer el contenido del archivo JSON
                    with open(original_json_path, 'r', encoding='utf-8') as f:
                        json_content = f.read()
                    
                    # Escribir el contenido en la ubicación correcta
                    with open(target_json_path, 'w', encoding='utf-8') as f:
                        f.write(json_content)
                        
                    logger.info(f"Archivo de metadatos movido a: {target_json_path}")
                except Exception as e:
                    logger.error(f"Error al mover el archivo de metadatos: {str(e)}")
            
            # 3. Mover las imágenes si existen
            nested_images_dir = nested_dir / "images"
            if nested_images_dir.exists() and nested_images_dir.is_dir():
                output_images_dir = output_subdir / "images"
                output_images_dir.mkdir(exist_ok=True)
                
                try:
                    # Mover todas las imágenes
                    for img_file in nested_images_dir.glob("*"):
                        target_img_path = output_images_dir / img_file.name
                        with open(img_file, 'rb') as src, open(target_img_path, 'wb') as dst:
                            dst.write(src.read())
                    logger.info(f"Imágenes movidas a: {output_images_dir}")
                except Exception as e:
                    logger.error(f"Error al mover imágenes: {str(e)}")
            
            # 4. Eliminar completamente la carpeta anidada para evitar confusiones
            try:
                import shutil
                shutil.rmtree(nested_dir)
                logger.info(f"Carpeta anidada eliminada: {nested_dir}")
            except Exception as e:
                logger.error(f"Error al eliminar carpeta anidada: {str(e)}")
        else:
            # Verificar si el archivo ya está en la carpeta principal
            if not markdown_path.exists():
                possible_files = list(output_subdir.glob("*.md"))
                if possible_files:
                    markdown_path = possible_files[0]
                    logger.info(f"Archivo Markdown encontrado: {markdown_path}")
                else:
                    logger.error(f"No se encontraron archivos Markdown en: {output_subdir}")
                    return None
        
        # Leer el archivo Markdown generado
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
                
            if not markdown_text:
                logger.error(f"El archivo Markdown está vacío: {markdown_path}")
                return None
                
            logger.info(f"Archivo Markdown leído exitosamente: {len(markdown_text)} caracteres")
        except Exception as e:
            logger.error(f"Error al leer el archivo Markdown: {str(e)}")
            return None
        
        # Dividir el contenido en chunks
        chunks = self.split_into_chunks(markdown_text, pdf_path.name)
        
        # Prepare los metadatos del documento
        word_count = len(markdown_text.split())
        
        return {
            'filename': pdf_path.name,
            'markdown_file': str(markdown_path),
            'num_chunks': len(chunks),
            'total_words': word_count,
            'chunks': chunks
        }
    
    def split_into_chunks(self, text: str, filename: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
        """
        Divide el texto Markdown en chunks respetando la estructura del documento.
        
        Args:
            text (str): Texto en formato Markdown
            filename (str): Nombre del archivo original
            chunk_size (int): Tamaño objetivo de cada chunk
            overlap (int): Cantidad de palabras de solapamiento
            
        Returns:
            List[str]: Lista de chunks
        """
        if not text:
            return []
        
        chunks = []
        lines = text.split('\n')
        
        # Detectar patrones markdown específicos
        title_pattern = re.compile(r'^#\s+')  # Título principal
        section_pattern = re.compile(r'^##\s+')  # Secciones
        article_pattern = re.compile(r'^###\s+Art(?:ículo|\.)\s*(\d+\º?)\.?', re.IGNORECASE)  # Artículos
        page_break_pattern = re.compile(r'^\d+\s*\n-{3,}$')  # Saltos de página generados por marker
        
        # Primero identificar las secciones principales y los artículos
        section_indices = []
        article_indices = []
        page_breaks = []
        
        for i, line in enumerate(lines):
            if title_pattern.match(line):
                section_indices.append(i)
            elif section_pattern.match(line):
                section_indices.append(i)
            elif article_pattern.match(line):
                article_indices.append(i)
            elif page_break_pattern.match(line) and i < len(lines) - 1:
                page_breaks.append(i)
        
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
        
        # Si no hay artículos ni secciones, dividir por páginas (si están marcadas)
        elif page_breaks:
            page_breaks = [-1] + page_breaks + [len(lines)]  # Añadir inicio y fin
            
            for i in range(len(page_breaks) - 1):
                start_idx = page_breaks[i] + 1
                end_idx = page_breaks[i + 1]
                
                page_lines = lines[start_idx:end_idx]
                page_text = '\n'.join(page_lines)
                
                # Dividir texto por párrafos si la página es muy grande
                if len(page_text.split()) > chunk_size:
                    paragraphs = re.split(r'\n\s*\n', page_text)
                    current_chunk = []
                    current_size = 0
                    
                    for para in paragraphs:
                        para_words = len(para.split())
                        
                        if current_size + para_words <= chunk_size:
                            current_chunk.append(para)
                            current_size += para_words
                        else:
                            # Guardar chunk actual
                            if current_chunk:
                                doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                                chunks.append(doc_prefix + '\n\n'.join(current_chunk))
                            
                            # Iniciar nuevo chunk
                            current_chunk = [para]
                            current_size = para_words
                    
                    # Guardar el último chunk
                    if current_chunk:
                        doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                        chunks.append(doc_prefix + '\n\n'.join(current_chunk))
                else:
                    doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                    chunks.append(doc_prefix + page_text)
        
        # Si no hay ninguna estructura detectable, usar división básica por párrafos
        else:
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
        """Procesa todos los documentos PDF en el directorio raw."""
        pdf_files = list(self.raw_dir.glob('**/*.pdf'))
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF")
        
        if not pdf_files:
            logger.warning(f"No se encontraron archivos PDF en {self.raw_dir}")
            return
        
        # Limpiar directorios existentes para comenzar desde cero
        for doc_dir in self.processed_dir.glob("*"):
            if doc_dir.is_dir() and doc_dir.name != "chunks" and doc_dir.name != "images":
                # Comprobar si corresponde a algún PDF que vamos a procesar
                if any(pdf.stem == doc_dir.name for pdf in pdf_files):
                    try:
                        import shutil
                        shutil.rmtree(doc_dir)
                        logger.info(f"Carpeta existente eliminada para reprocesamiento: {doc_dir}")
                    except Exception as e:
                        logger.error(f"Error al eliminar carpeta: {doc_dir}, {str(e)}")
        
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
    """Función principal para ejecutar el preprocesamiento con Marker."""
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    logger.info(f"Iniciando preprocesamiento con Marker. Directorio raw: {raw_dir}, Directorio procesado: {processed_dir}")
    
    # Usar el preprocesador basado en Marker
    preprocessor = MarkerPreprocessor(raw_dir, processed_dir)
    preprocessor.process_all_documents()

if __name__ == "__main__":
    main() 