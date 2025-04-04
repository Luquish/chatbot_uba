"""
Document preprocessor for the UBA Medicina chatbot RAG system.
Extracts and processes text from PDF documents to generate embeddings.

This module implements:
1. Text extraction from PDFs using pdfminer
2. Text cleaning and normalization
3. Chunk segmentation with overlap
4. Document structure detection
5. Metadata generation

Key features:
- Robust PDF extraction
- Document structure preservation
- Different content type handling
- Detailed logging
- Data validation
"""

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

# Detailed logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    """
    Preprocesses PDF documents for the RAG system.
    
    Responsibilities:
    - PDF text extraction
    - Text cleaning and normalization
    - Chunk segmentation
    - Structure detection
    - Metadata generation
    """
    
    def __init__(self, raw_dir: str, processed_dir: str):
        """
        Initializes the document preprocessor.
        
        Args:
            raw_dir (str): Directory with unprocessed documents
            processed_dir (str): Directory to save processed documents
            
        Notes:
        - Creates necessary directories
        - Configures logging
        - Prepares file structure
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extracts text from a PDF file using pdfminer.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
            
        Notes:
        - Uses multiple extraction methods
        - Handles errors and special cases
        - Validates extracted text quality
        """
        try:
            logger.info(f"Extracting text from: {pdf_path}")
            
            # Method 1: Use extract_text (simpler but sometimes less accurate)
            text = extract_text(pdf_path)
            
            # If text is empty or too short, use alternative method
            if not text or len(text) < 100:
                logger.warning(f"First extraction produced insufficient text, using alternative method for {pdf_path}")
                
                # Method 2: More detailed using pdfminer components
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
            
            # Basic verification
            if not text:
                logger.error(f"Could not extract text from {pdf_path}")
                return ""
                
            logger.info(f"Text extracted successfully: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
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
        text = re.sub(r'\s+', ' ', text)
        
        # Reemplazar múltiples saltos de línea por uno solo
        text = re.sub(r'\n+', '\n', text)
        
        # Eliminar espacios alrededor de cada línea
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Eliminar líneas que solo contengan números o sean muy cortas (menos de 3 caracteres)
        lines = [line for line in text.split('\n') if len(line) > 3 and not line.strip().isdigit()]
        text = '\n'.join(lines)
        
        return text.strip()

    def split_into_chunks(self, text: str, filename: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Divide el texto en chunks con solapamiento para preservar el contexto.
        Utiliza párrafos como delimitadores naturales cuando es posible.
        
        Args:
            text (str): Texto a dividir
            filename (str): Nombre del archivo (para mejorar chunking basado en tipo de documento)
            chunk_size (int): Tamaño objetivo de cada chunk
            overlap (int): Cantidad de palabras de solapamiento entre chunks
            
        Returns:
            List[str]: Lista de chunks
        """
        if not text:
            return []
            
        # Dividir por párrafos (separados por líneas en blanco)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        logger.info(f"Documento dividido en {len(paragraphs)} párrafos")
        
        # Si tenemos artículos numerados (como en reglamentos), detectarlos
        is_article_based = False
        article_pattern = re.compile(r'artículo\s+\d+', re.IGNORECASE)
        
        # Verificar si el documento está basado en artículos
        article_count = sum(1 for p in paragraphs if article_pattern.search(p))
        if article_count > 2 or 'reglamento' in filename.lower() or 'condiciones' in filename.lower():
            is_article_based = True
            logger.info(f"Documento basado en artículos detectado: {filename}")
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Función para agregar un chunk completo
        def add_chunk():
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                # Incluir información del archivo en el chunk
                doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                chunks.append(doc_prefix + chunk_text)
                logger.debug(f"Chunk creado: {len(chunk_text)} caracteres")
        
        # Si el documento está basado en artículos, intentar mantener artículos juntos
        if is_article_based:
            current_article = None
            article_chunks = []
            
            for paragraph in paragraphs:
                # Comprobar si este párrafo comienza un nuevo artículo
                article_match = article_pattern.search(paragraph)
                
                if article_match:
                    # Si teníamos un artículo anterior, lo agregamos como chunk
                    if current_article:
                        article_chunks.append(current_article)
                    current_article = paragraph
                elif current_article:
                    # Añadir al artículo actual
                    current_article += " " + paragraph
                else:
                    # Párrafo que no pertenece a ningún artículo
                    article_chunks.append(paragraph)
            
            # Añadir el último artículo
            if current_article:
                article_chunks.append(current_article)
            
            # Procesar los chunks basados en artículos
            for article_text in article_chunks:
                words = article_text.split()
                if len(words) <= chunk_size * 1.5:  # Si es menos de 1.5 veces el tamaño objetivo
                    doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                    chunks.append(doc_prefix + article_text)
                else:
                    # El artículo es demasiado largo, dividirlo respetando el tamaño de chunk
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = ' '.join(words[i:i + chunk_size])
                        doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                        chunks.append(doc_prefix + chunk)
        else:
            # Método estándar basado en palabras con overlap
            words = text.split()
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                doc_prefix = f"[Documento: {os.path.basename(filename)}] "
                chunks.append(doc_prefix + chunk)
        
        # Asegurar que los chunks no sean demasiado pequeños
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) >= 50:  # Mínimo 50 palabras por chunk
                final_chunks.append(chunk)
            else:
                # Si es muy pequeño y hay chunks previos, agregarlo al anterior
                if final_chunks:
                    final_chunks[-1] += " " + chunk
                else:
                    final_chunks.append(chunk)
        
        logger.info(f"Documento dividido en {len(final_chunks)} chunks")
        return final_chunks

    def process_document(self, pdf_path: Path) -> Dict:
        """
        Procesa un documento PDF completo con mejor logging e información.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            Dict: Diccionario con metadatos y chunks del documento
        """
        logger.info(f"Procesando documento: {pdf_path}")
        
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            logger.error(f"No se pudo extraer texto de {pdf_path}")
            return None
            
        # Loguear primeros 100 caracteres para verificación
        logger.info(f"Muestra de texto extraído: {text[:100]}...")
        
        clean_text = self.clean_text(text)
        logger.info(f"Texto limpiado: {len(clean_text)} caracteres")
        
        # Loguear primeros 100 caracteres de texto limpio para verificación
        logger.info(f"Muestra de texto limpio: {clean_text[:100]}...")
        
        chunks = self.split_into_chunks(clean_text, pdf_path.name)
        
        word_count = len(clean_text.split())
        logger.info(f"Documento procesado: {word_count} palabras, {len(chunks)} chunks")
        
        return {
            'filename': pdf_path.name,
            'num_chunks': len(chunks),
            'total_words': word_count,
            'chunks': chunks
        }

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