import os
import logging
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import faiss
import json
import ast
from tqdm import tqdm
from dotenv import load_dotenv
from models.openai_model import OpenAIEmbedding

# Cargar variables de entorno de forma más explícita
dotenv_path = Path('.env')
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

logger.info(f"OPENAI_API_KEY configurada: {'Sí' if OPENAI_API_KEY else 'No'}")
logger.info(f"Modelo de embeddings: {OPENAI_EMBEDDING_MODEL}")

class EmbeddingGenerator:
    def __init__(self, processed_dir: str, embeddings_dir: str):
        """
        Inicializa el generador de embeddings. Siempre usa OpenAI.
        
        Args:
            processed_dir (str): Directorio con documentos procesados
            embeddings_dir (str): Directorio para guardar embeddings
        """
        self.processed_dir = Path(processed_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar OpenAI
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY no está configurada. Es necesaria para generar embeddings.")
        
        # Inicializar modelo de OpenAI
        logger.info(f"Inicializando modelo de embeddings OpenAI: {OPENAI_EMBEDDING_MODEL}")
        self.model = OpenAIEmbedding(
            model_name=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
            timeout=30
        )
        logger.info(f"Dimensión del modelo OpenAI: {self.model.get_sentence_embedding_dimension()}")
        logger.info("OpenAI inicializado correctamente para embeddings")
        
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
        Genera embeddings para una lista de textos usando OpenAI.
        
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
        logger.info(f"Generando embeddings con OpenAI para {total_texts} textos")
        
        # Preprocesar textos 
        valid_texts = []
        invalid_indices = []
        
        for i, text in enumerate(texts):
            if not text or not isinstance(text, str):
                logger.warning(f"Texto inválido en índice {i}")
                invalid_indices.append(i)
                valid_texts.append("texto inválido")
            else:
                # Limpiar y preparar el texto
                text = text.strip()
                if len(text) < 10:
                    logger.warning(f"Texto muy corto en índice {i}")
                    text = text + " " + text  # Duplicar texto corto
                valid_texts.append(text)
        
        # Generar embeddings con OpenAI
        embeddings = []
        # OpenAI soporta batches más grandes, dividimos en bloques de 100 para seguridad
        openai_batch_size = min(batch_size * 4, 100)
        
        try:
            for i in tqdm(range(0, len(valid_texts), openai_batch_size), desc="Generando embeddings con OpenAI"):
                batch = valid_texts[i:i + openai_batch_size]
                try:
                    # Llamar a la API de OpenAI
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=False
                    )
                    embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error en batch OpenAI {i}: {str(e)}")
                    raise
            
            # Concatenar embeddings
            all_embeddings = np.vstack(embeddings)
            
            # Normalizar si es necesario
            if np.any(np.sum(all_embeddings * all_embeddings, axis=1) > 1.0):
                logger.info("Normalizando embeddings finales...")
                all_embeddings = all_embeddings / np.sqrt(np.sum(all_embeddings * all_embeddings, axis=1, keepdims=True))
            
            return all_embeddings
                
        except Exception as e:
            logger.error(f"Error general en generación de embeddings con OpenAI: {str(e)}")
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
        """Procesa todos los documentos y genera embeddings con OpenAI."""
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
            
            # Generar embeddings usando OpenAI
            logger.info(f"Generando embeddings usando OpenAI")
            embeddings = self.generate_embeddings(texts)
            
            if embeddings.size == 0:
                raise ValueError("No se generaron embeddings (array vacío)")
            
            # Crear y guardar índice FAISS
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
                'embedding_model': 'OpenAI',
                'model_name': OPENAI_EMBEDDING_MODEL,
                'dimension': self.model.get_sentence_embedding_dimension(),
                'num_vectors': len(texts),
                'environment': 'development',
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
    
    # Mostrar configuración
    logger.info("=== Configuración de embeddings ===")
    
    if OPENAI_API_KEY:
        logger.info(f"Modelo: OpenAI {OPENAI_EMBEDDING_MODEL}")
        logger.info("API Key de OpenAI encontrada ✓")
    else:
        logger.error("❌ API Key de OpenAI NO encontrada - Se requiere para el funcionamiento")
        logger.error("El script no puede continuar sin una API Key de OpenAI")
        return
    
    logger.info("================================")
    logger.info(f"Iniciando generación de embeddings. Dir procesado: {processed_dir}, Dir embeddings: {embeddings_dir}")
    
    try:
        generator = EmbeddingGenerator(processed_dir, embeddings_dir)
        generator.process_documents()
        logger.info("Proceso de generación de embeddings completado exitosamente")
    except Exception as e:
        logger.error(f"Error en la generación de embeddings: {str(e)}")
        logger.error("El proceso no pudo completarse correctamente")

if __name__ == "__main__":
    main() 