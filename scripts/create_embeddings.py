import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import faiss
import json
import ast
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
        
        # Inicializar el modelo de embeddings adecuado para español
        embedding_model_options = [
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # Buen modelo multilingüe
            'hiiamsid/sentence_similarity_spanish_es',  # Especializado en español
            'intfloat/multilingual-e5-large'  # Modelo más grande y preciso
        ]
        
        # Por defecto usar el primer modelo, pero permitir configuración
        embedding_model_name = os.getenv('EMBEDDING_MODEL', embedding_model_options[0])
        logger.info(f"Usando modelo de embeddings: {embedding_model_name}")
        
        try:
            self.model = SentenceTransformer(embedding_model_name)
            logger.info(f"Modelo de embeddings inicializado: {embedding_model_name}")
            logger.info(f"Dimensión del modelo: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error al cargar modelo de embeddings: {str(e)}")
            # Fallback al modelo más básico si hay error
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Usando modelo de fallback")
        
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
        Carga los documentos procesados desde el CSV manejando posibles errores.
        
        Returns:
            pd.DataFrame: DataFrame con los documentos procesados
        """
        csv_path = self.processed_dir / 'processed_documents.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo {csv_path}")
        
        try:
            # Leer el DataFrame asegurando que las columnas sean correctas
            df = pd.read_csv(csv_path)
            required_columns = ['filename', 'chunks']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Columna requerida '{col}' no encontrada en el CSV")
            
            # Convertir las columnas de tipo lista desde strings
            if 'chunks' in df.columns and isinstance(df['chunks'].iloc[0], str):
                logger.info("Convirtiendo columna 'chunks' de string a lista...")
                try:
                    # Intento 1: Usar ast.literal_eval para evaluar la string como lista
                    df['chunks'] = df['chunks'].apply(lambda x: ast.literal_eval(x))
                except:
                    try:
                        # Intento 2: Formato específico
                        df['chunks'] = df['chunks'].apply(lambda x: x.strip('[]').split('", "'))
                    except:
                        # Intento 3: Tratarlo como JSON
                        df['chunks'] = df['chunks'].apply(lambda x: json.loads(x.replace("'", '"')))
            
            logger.info(f"CSV cargado correctamente. {len(df)} documentos encontrados.")
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar el CSV: {str(e)}")
            raise
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Genera embeddings para una lista de textos con más robustez y control.
        
        Args:
            texts (List[str]): Lista de textos a procesar
            batch_size (int): Tamaño del batch para procesamiento
            
        Returns:
            np.ndarray: Array de embeddings
        """
        if not texts:
            logger.warning("Lista de textos vacía, no se generarán embeddings")
            return np.array([])
        
        total_texts = len(texts)
        logger.info(f"Generando embeddings para {total_texts} textos con batch_size={batch_size}")
        
        embeddings = []
        
        # Verificar textos vacíos o inválidos
        valid_texts = []
        invalid_indices = []
        
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str) or len(text.strip()) < 10:
                logger.warning(f"Texto inválido en índice {i}: '{text[:20]}...'")
                invalid_indices.append(i)
                # Usar un texto por defecto para evitar errores
                valid_texts.append("texto vacío o inválido")
            else:
                valid_texts.append(text)
        
        if invalid_indices:
            logger.warning(f"Se encontraron {len(invalid_indices)} textos inválidos de {total_texts}")
        
        # Procesar en batches
        try:
            for i in tqdm(range(0, len(valid_texts), batch_size), desc="Generando embeddings"):
                batch = valid_texts[i:i + batch_size]
                
                # Limitar longitud de textos para evitar errores con tokens muy largos
                batch = [text[:8192] if len(text) > 8192 else text for text in batch]
                
                try:
                    batch_embeddings = self.model.encode(batch, show_progress_bar=False, 
                                                    convert_to_numpy=True, normalize_embeddings=True)
                    embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error al generar embeddings para batch {i}: {str(e)}")
                    # Crear embeddings vacíos para este batch si falla
                    dummy_embeddings = np.zeros((len(batch), self.model.get_sentence_embedding_dimension()))
                    embeddings.append(dummy_embeddings)
                    
            # Unir todos los embeddings
            all_embeddings = np.vstack(embeddings)
            logger.info(f"Embeddings generados: {all_embeddings.shape}")
            
            # Verificar si hay NaNs y reemplazarlos
            if np.isnan(all_embeddings).any():
                logger.warning("Se detectaron NaNs en los embeddings. Reemplazando con ceros.")
                all_embeddings = np.nan_to_num(all_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error general al generar embeddings: {str(e)}")
            raise
        
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Crea un índice FAISS para los embeddings más optimizado.
        
        Args:
            embeddings (np.ndarray): Array de embeddings
            
        Returns:
            faiss.Index: Índice FAISS
        """
        dimension = embeddings.shape[1]
        logger.info(f"Creando índice FAISS con {embeddings.shape[0]} vectores de dimensión {dimension}")
        
        # Asegurar que los embeddings sean float32 para FAISS
        embeddings = embeddings.astype('float32')
        
        # Verificar si tenemos suficientes vectores para usar índices más avanzados
        if embeddings.shape[0] > 10000:
            # Para colecciones grandes, usar un índice IVF para búsqueda más rápida
            nlist = min(int(np.sqrt(embeddings.shape[0])), 100)  # Número de clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            
            # Necesita entrenamiento
            logger.info(f"Entrenando índice IVFFlat con {nlist} clusters...")
            index.train(embeddings)
            index.add(embeddings)
            logger.info("Índice IVFFlat creado y entrenado")
        else:
            # Para colecciones pequeñas, usar un índice plano (más simple pero preciso)
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            logger.info("Índice FlatL2 creado")
        
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
        Guarda metadatos de los embeddings con más información.
        
        Args:
            texts (List[str]): Lista de textos
            filenames (List[str]): Lista de nombres de archivo
            chunk_indices (List[int]): Lista de índices de chunks
        """
        # Crear DataFrame con información adicional
        metadata = pd.DataFrame({
            'text': texts,
            'filename': filenames,
            'chunk_index': chunk_indices,
            'text_length': [len(text) for text in texts],
            'word_count': [len(text.split()) for text in texts]
        })
        
        # Añadir timestamp para seguimiento
        from datetime import datetime
        metadata['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Guardar en CSV
        metadata_path = self.embeddings_dir / 'metadata.csv'
        metadata.to_csv(metadata_path, index=False)
        logger.info(f"Metadatos guardados en {metadata_path}")
        
        # Guardar también un resumen por archivo
        summary = metadata.groupby('filename').agg({
            'chunk_index': 'count',
            'text_length': ['sum', 'mean'],
            'word_count': ['sum', 'mean']
        }).reset_index()
        summary.columns = ['filename', 'num_chunks', 'total_chars', 'avg_chars_per_chunk', 
                           'total_words', 'avg_words_per_chunk']
        
        summary_path = self.embeddings_dir / 'metadata_summary.csv'
        summary.to_csv(summary_path, index=False)
        logger.info(f"Resumen por archivo guardado en {summary_path}")
        
    def process_documents(self):
        """Procesa todos los documentos y genera embeddings con mejor manejo de errores."""
        try:
            # Cargar documentos procesados
            df = self.load_processed_documents()
            logger.info(f"Documentos cargados: {len(df)}")
            
            # Preparar datos para embeddings
            texts = []
            filenames = []
            chunk_indices = []
            
            # Contar chunks totales para información
            total_chunks = 0
            for _, row in df.iterrows():
                chunks = row['chunks']
                if isinstance(chunks, list):
                    total_chunks += len(chunks)
            
            logger.info(f"Total de chunks a procesar: {total_chunks}")
            
            for _, row in df.iterrows():
                filename = row['filename']
                chunks = row['chunks']
                
                # Verificar tipo de chunks (debería ser lista)
                if not isinstance(chunks, list):
                    logger.warning(f"Formato incorrecto para chunks en {filename}, intentando convertir...")
                    try:
                        if isinstance(chunks, str):
                            chunks = ast.literal_eval(chunks)
                        else:
                            logger.error(f"No se puede procesar chunks para {filename}, tipo: {type(chunks)}")
                            continue
                    except Exception as e:
                        logger.error(f"Error al convertir chunks para {filename}: {str(e)}")
                        continue
                
                logger.info(f"Procesando {len(chunks)} chunks de {filename}")
                
                for i, chunk in enumerate(chunks):
                    texts.append(chunk)
                    filenames.append(filename)
                    chunk_indices.append(i)
                
            if not texts:
                raise ValueError("No se encontraron textos para procesar")
                
            logger.info(f"Datos preparados: {len(texts)} textos de {len(df)} documentos")
            
            # Generar embeddings
            embeddings = self.generate_embeddings(texts)
            
            if embeddings.size == 0:
                raise ValueError("No se generaron embeddings (array vacío)")
            
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
            
            # Guardar configuración utilizada para referencia
            config = {
                'date': pd.Timestamp.now().isoformat(),
                'embedding_model': self.model.get_sentence_embedding_dimension(),
                'dimension': self.model.get_sentence_embedding_dimension(),
                'num_vectors': len(texts),
                'environment': ENVIRONMENT,
                'documents': [f for f in df['filename']]
            }
            
            with open(self.embeddings_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info("Proceso completado exitosamente")
            
        except Exception as e:
            logger.error(f"Error en process_documents: {str(e)}", exc_info=True)
            raise

def main():
    """Función principal para ejecutar la generación de embeddings."""
    processed_dir = Path('data/processed')
    embeddings_dir = Path('data/embeddings')
    
    # Verificar que exista el directorio de documentos procesados
    if not processed_dir.exists():
        logger.error(f"No existe el directorio de documentos procesados: {processed_dir}")
        logger.info("Por favor, ejecute primero el script de preprocesamiento.")
        return
    
    logger.info(f"Iniciando generación de embeddings. Dir procesado: {processed_dir}, Dir embeddings: {embeddings_dir}")
    
    generator = EmbeddingGenerator(processed_dir, embeddings_dir)
    generator.process_documents()

if __name__ == "__main__":
    main() 