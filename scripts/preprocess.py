import os
import logging
from pathlib import Path
from typing import List, Dict
import PyPDF2
from pdfminer.high_level import extract_text
import pandas as pd
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        Extrae texto de un archivo PDF usando pdfminer.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            str: Texto extraído del PDF
        """
        try:
            return extract_text(pdf_path)
        except Exception as e:
            logger.error(f"Error al extraer texto de {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Limpia el texto extraído.
        
        Args:
            text (str): Texto a limpiar
            
        Returns:
            str: Texto limpio
        """
        # Eliminar espacios múltiples
        text = ' '.join(text.split())
        # Eliminar caracteres especiales
        text = ''.join(char for char in text if char.isprintable())
        return text.strip()

    def split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Divide el texto en chunks de tamaño fijo.
        
        Args:
            text (str): Texto a dividir
            chunk_size (int): Tamaño de cada chunk
            
        Returns:
            List[str]: Lista de chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 por el espacio
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def process_document(self, pdf_path: Path) -> Dict:
        """
        Procesa un documento PDF completo.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            Dict: Diccionario con metadatos y chunks del documento
        """
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
            
        clean_text = self.clean_text(text)
        chunks = self.split_into_chunks(clean_text)
        
        return {
            'filename': pdf_path.name,
            'num_chunks': len(chunks),
            'total_words': len(clean_text.split()),
            'chunks': chunks
        }

    def process_all_documents(self):
        """Procesa todos los documentos PDF en el directorio raw."""
        pdf_files = list(self.raw_dir.glob('**/*.pdf'))
        logger.info(f"Encontrados {len(pdf_files)} archivos PDF")
        
        processed_data = []
        for pdf_path in tqdm(pdf_files, desc="Procesando documentos"):
            result = self.process_document(pdf_path)
            if result:
                processed_data.append(result)
                
        # Guardar resultados en CSV
        if processed_data:
            df = pd.DataFrame(processed_data)
            output_path = self.processed_dir / 'processed_documents.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Datos procesados guardados en {output_path}")
            
            # Guardar chunks individuales
            chunks_dir = self.processed_dir / 'chunks'
            chunks_dir.mkdir(exist_ok=True)
            
            for doc in processed_data:
                for i, chunk in enumerate(doc['chunks']):
                    chunk_path = chunks_dir / f"{doc['filename']}_chunk_{i}.txt"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)

def main():
    """Función principal para ejecutar el preprocesamiento."""
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    preprocessor = DocumentPreprocessor(raw_dir, processed_dir)
    preprocessor.process_all_documents()

if __name__ == "__main__":
    main() 