import os
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determinar el entorno (desarrollo o producción)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
logger.info(f"Iniciando generación de embeddings en entorno: {ENVIRONMENT}")

# Importaciones condicionales para Pinecone (solo en producción)
if ENVIRONMENT == 'production':
    try:
        import pinecone
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
        PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'uba-chatbot-embeddings')
        if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
            logger.warning("Falta configuración de Pinecone. Se usará FAISS local.")
            ENVIRONMENT = 'development'
        else:
            logger.info("Usando Pinecone para almacenamiento de embeddings en producción.")
    except ImportError:
        logger.warning("No se pudo importar pinecone. Se usará FAISS local.")
        ENVIRONMENT = 'development'

class EmbeddingGenerator:
    def __init__(self, processed_dir: str, embeddings_dir: str):
        """
        Inicializa el generador de embeddings.
        
        Args:
            processed_dir (str): Directorio con documentos procesados
            embeddings_dir (str): Directorio para guardar embeddings
        """
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar el modelo de embeddings
        embedding_model_name = os.getenv('EMBEDDING_MODEL', 
                                        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.model = SentenceTransformer(embedding_model_name)
        self.is_production = ENVIRONMENT == 'production'
        
        # Inicializar Pinecone en entorno de producción
        if self.is_production:
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            
            # Verificar si el índice existe, si no, crearlo
            if PINECONE_INDEX_NAME not in pinecone.list_indexes():
                logger.info(f"Creando índice '{PINECONE_INDEX_NAME}' en Pinecone...")
                pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=self.model.get_sentence_embedding_dimension(),
                    metric="cosine"
                )
            
            self.pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
            logger.info(f"Índice Pinecone inicializado: {PINECONE_INDEX_NAME}")
        
    def load_processed_documents(self) -> pd.DataFrame:
        """
        Carga los documentos procesados desde el CSV.
        
        Returns:
            pd.DataFrame: DataFrame con los documentos procesados
        """
        csv_path = self.processed_dir / 'processed_documents.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo {csv_path}")
            
        return pd.read_csv(csv_path)
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts (List[str]): Lista de textos a procesar
            batch_size (int): Tamaño del batch para procesamiento
            
        Returns:
            np.ndarray: Array de embeddings
        """
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generando embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
        
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Crea un índice FAISS para los embeddings.
        
        Args:
            embeddings (np.ndarray): Array de embeddings
            
        Returns:
            faiss.Index: Índice FAISS
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index
    
    def store_in_pinecone(self, texts: List[str], embeddings: np.ndarray, 
                           filenames: List[str], chunk_indices: List[int]) -> None:
        """
        Almacena los embeddings en Pinecone.
        
        Args:
            texts (List[str]): Lista de textos
            embeddings (np.ndarray): Array de embeddings
            filenames (List[str]): Lista de nombres de archivo
            chunk_indices (List[int]): Lista de índices de chunks
        """
        # Preparar los vectores para la inserción en Pinecone
        vectors_to_upsert = []
        
        for i, (text, embedding, filename, chunk_idx) in enumerate(
            zip(texts, embeddings, filenames, chunk_indices)
        ):
            # Crear un ID único para cada vector
            vector_id = f"{filename.replace('.', '_')}_{chunk_idx}"
            
            # Crear el registro con el vector y metadatos
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding.tolist(),
                'metadata': {
                    'text': text,
                    'filename': filename,
                    'chunk_index': chunk_idx
                }
            })
            
            # Insertar en lotes para mejorar el rendimiento
            if len(vectors_to_upsert) >= 100 or i == len(texts) - 1:
                logger.info(f"Insertando lote de {len(vectors_to_upsert)} vectores en Pinecone...")
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
                
        logger.info(f"Embeddings almacenados en Pinecone. Total: {len(texts)} vectores.")
        
    def save_metadata(self, texts: List[str], filenames: List[str], chunk_indices: List[int]):
        """
        Guarda metadatos de los embeddings.
        
        Args:
            texts (List[str]): Lista de textos
            filenames (List[str]): Lista de nombres de archivo
            chunk_indices (List[int]): Lista de índices de chunks
        """
        metadata = pd.DataFrame({
            'text': texts,
            'filename': filenames,
            'chunk_index': chunk_indices
        })
        metadata_path = self.embeddings_dir / 'metadata.csv'
        metadata.to_csv(metadata_path, index=False)
        
    def process_documents(self):
        """Procesa todos los documentos y genera embeddings."""
        # Cargar documentos procesados
        df = self.load_processed_documents()
        
        # Preparar datos para embeddings
        texts = []
        filenames = []
        chunk_indices = []
        
        for _, row in df.iterrows():
            for i, chunk in enumerate(row['chunks']):
                texts.append(chunk)
                filenames.append(row['filename'])
                chunk_indices.append(i)
                
        # Generar embeddings
        embeddings = self.generate_embeddings(texts)
        
        if self.is_production:
            # Almacenar en Pinecone para producción
            self.store_in_pinecone(texts, embeddings, filenames, chunk_indices)
            logger.info(f"Embeddings almacenados en Pinecone ({PINECONE_INDEX_NAME}).")
        else:
            # Crear y guardar índice FAISS para desarrollo
            index = self.create_faiss_index(embeddings)
            index_path = self.embeddings_dir / 'faiss_index.bin'
            faiss.write_index(index, str(index_path))
            logger.info(f"Índice FAISS guardado en {index_path}")
        
        # Guardar metadatos (útil para ambos entornos)
        self.save_metadata(texts, filenames, chunk_indices)
        
        logger.info(f"Total de chunks procesados: {len(texts)}")
        logger.info(f"Dimensión de los embeddings: {embeddings.shape}")

def main():
    """Función principal para ejecutar la generación de embeddings."""
    processed_dir = Path('data/processed')
    embeddings_dir = Path('data/embeddings')
    
    generator = EmbeddingGenerator(processed_dir, embeddings_dir)
    generator.process_documents()

if __name__ == "__main__":
    main() 