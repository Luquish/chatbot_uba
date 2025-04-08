import os
import logging
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import torch
import faiss
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
import huggingface_hub
import psutil  # Añadido para verificar la memoria disponible
import re

# Crear directorio de logs si no existe
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Lista de palabras de saludo
greeting_words = ['hola', 'buenas', 'buen día', 'buen dia', 'buenos días', 'buenos dias', 
                'buenas tardes', 'buenas noches', 'saludos', 'que tal', 'qué tal', 'como va', 'cómo va']

# Lista de emojis para enriquecer las respuestas
information_emojis = ["📚", "📖", "ℹ️", "📊", "🔍", "📝", "📋", "📈", "📌", "🧠"]
greeting_emojis = ["👋", "😊", "🤓", "👨‍⚕️", "👩‍⚕️", "🎓", "🌟"]

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Salida a consola
        logging.FileHandler(log_dir / 'app.log')  # Salida a archivo
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configurar token de Hugging Face si está disponible
if "HUGGING_FACE_HUB_TOKEN" in os.environ:
    huggingface_hub.login(token=os.environ["HUGGING_FACE_HUB_TOKEN"], add_to_git_credential=False)
    logger.info("Configurado token de Hugging Face desde variables de entorno")

# Determinar el entorno (desarrollo o producción)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
logger.info(f"Iniciando sistema RAG en entorno: {ENVIRONMENT}")

# Función para configurar el dispositivo según preferencias
def get_device():
    """Configura el dispositivo según preferencias y disponibilidad."""
    device_pref = os.getenv('DEVICE', 'auto')
    
    if device_pref == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_pref  # Usa específicamente cuda, cpu o mps si se especifica
    
    logger.info(f"Usando dispositivo: {device}")
    return device

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
            logger.info("Usando Pinecone para búsqueda de embeddings en producción.")
            # Inicializar pinecone
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    except ImportError:
        logger.warning("No se pudo importar pinecone. Se usará FAISS local.")
        ENVIRONMENT = 'development'

# Clase base abstracta para búsquedas vectoriales
class VectorStore:
    def search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """Búsqueda de vectores similares"""
        raise NotImplementedError("Este método debe ser implementado por las subclases")

# Implementación para FAISS (desarrollo)
class FAISSVectorStore(VectorStore):
    def __init__(self, index_path: str, metadata_path: str):
        """
        Inicializa el almacén vectorial FAISS.
        
        Args:
            index_path (str): Ruta al archivo de índice FAISS
            metadata_path (str): Ruta al archivo de metadatos
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No se encontró el índice FAISS en {index_path}")
            
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)
        logger.info(f"Índice FAISS cargado con {self.index.ntotal} vectores")
    
        # Umbral de similitud mínimo (ajustable según necesidad)
        self.similarity_threshold = 0.1  # Reducido de 0.6 a 0.1 para ser más permisivo
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Búsqueda de vectores similares en FAISS con mejoras.
        
        Args:
            query_embedding (List[float]): Embedding de la consulta
            k (int): Número de resultados a retornar
            
        Returns:
            List[Dict]: Lista de resultados con metadatos
        """
        # Convertir a numpy y normalizar
        query_embedding_np = np.array(query_embedding).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding_np)
        
        # Realizar búsqueda
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Convertir distancias L2 a similitud coseno
        similarities = np.clip(1 - distances[0] / 2, 0, 1)  # Asegurar rango [0,1]
        
        # Filtrar por umbral y ordenar por similitud
        results = []
        seen_texts = set()  # Para evitar duplicados
        
        for idx, sim in zip(indices[0], similarities):
            if idx < 0 or idx >= len(self.metadata):  # Índice inválido
                continue
            
            metadata = self.metadata.iloc[idx].to_dict()
            text = metadata.get('text', '')
            
            # Evitar duplicados y textos muy cortos
            if (text in seen_texts or 
                len(text) < 50 or 
                sim < self.similarity_threshold):
                continue
                
            seen_texts.add(text)
            metadata['similarity'] = float(sim)
            metadata['distance'] = float(distances[0][indices[0].tolist().index(idx)])
            results.append(metadata)
                
        # Ordenar por similitud
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        
        # Logging de resultados
        if results:
            logger.info(f"Mejor similitud encontrada: {results[0]['similarity']:.3f}")
            logger.info(f"Documento más relevante: {results[0].get('filename', 'N/A')}")
            logger.info(f"Número de resultados únicos: {len(results)}")
        else:
            logger.warning("No se encontraron resultados que superen el umbral de similitud")
        
        return results[:k]  # Limitar a k resultados

# Implementación para Pinecone (producción)
class PineconeVectorStore(VectorStore):
    def __init__(self, index_name: str):
        """
        Inicializa el almacén vectorial Pinecone.
        
        Args:
            index_name (str): Nombre del índice en Pinecone
        """
        if index_name not in pinecone.list_indexes():
            raise ValueError(f"No se encontró el índice Pinecone '{index_name}'")
            
        self.index = pinecone.Index(index_name)
        stats = self.index.describe_index_stats()
        logger.info(f"Índice Pinecone '{index_name}' cargado con {stats['total_vector_count']} vectores")
    
    def search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Búsqueda de vectores similares en Pinecone.
        
        Args:
            query_embedding (List[float]): Embedding de la consulta
            k (int): Número de resultados a retornar
            
        Returns:
            List[Dict]: Lista de resultados con metadatos
        """
        # Realizar búsqueda en Pinecone
        query_results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Formatear resultados
        results = []
        for match in query_results['matches']:
            # Extraer metadatos
            metadata = match['metadata']
            metadata['distance'] = 1.0 - match['score']  # Convertir similitud coseno a distancia
            results.append(metadata)
                
        return results

def get_available_memory_gb():
    """
    Obtiene la cantidad de memoria RAM disponible en GB.
    
    Returns:
        float: Memoria disponible en GB
    """
    try:
        # Intentar obtener la memoria virtual disponible
        available = psutil.virtual_memory().available
        available_gb = available / (1024 ** 3)  # Convertir a GB
        logger.info(f"Memoria disponible: {available_gb:.2f} GB")
        return available_gb
    except Exception as e:
        logger.warning(f"Error al obtener memoria disponible: {str(e)}")
        # En caso de error, devolver un valor que permita cargar el modelo grande
        # para no cambiar el comportamiento previo
        return 32.0

def load_model_with_fallback(model_path: str, load_kwargs: Dict) -> tuple:
    """
    Intenta cargar un modelo y, si falla por autenticación o memoria insuficiente, 
    usa un modelo abierto como alternativa.
    Soporta optimizaciones según plataforma.
    
    Args:
        model_path (str): Ruta al modelo preferido
        load_kwargs (Dict): Argumentos para cargar el modelo
        
    Returns:
        tuple: (modelo, tokenizer, nombre_del_modelo_cargado)
    """
    # Modelo abierto de respaldo
    fallback_model = os.getenv('FALLBACK_MODEL_NAME', 'google/gemma-2b')
    
    # Detectar si estamos en Mac con Apple Silicon
    is_mac_silicon = torch.backends.mps.is_available()
    
    # Verificar memoria disponible
    available_memory_gb = get_available_memory_gb()
    memory_threshold_gb = 16.0  # Necesitamos al menos 16GB libres para Mistral 7B
    
    # Decidir si usar el modelo de respaldo basado en la memoria disponible
    if available_memory_gb < memory_threshold_gb:
        logger.warning(f"Memoria disponible ({available_memory_gb:.2f} GB) es menor que el umbral recomendado ({memory_threshold_gb} GB)")
        logger.warning(f"Usando modelo de respaldo para evitar errores de memoria: {fallback_model}")
        model_path = fallback_model
    
    # Ajustar configuración según la plataforma
    if is_mac_silicon:
        logger.info("Detectado Mac con Apple Silicon, ajustando configuración de carga")
        
        # Para Mac Silicon, desactivamos bitsandbytes independientemente de la configuración
        # Quitamos la cuantización que causa problemas en Mac
        if "load_in_8bit" in load_kwargs and load_kwargs["load_in_8bit"]:
            logger.warning("Desactivando cuantización de 8 bits en Apple Silicon")
            load_kwargs["load_in_8bit"] = False
        if "load_in_4bit" in load_kwargs and load_kwargs["load_in_4bit"]:
            logger.warning("Desactivando cuantización de 4 bits en Apple Silicon")
            load_kwargs["load_in_4bit"] = False
        
        # Configurar para usar la aceleración de MPS por defecto
        if "device_map" in load_kwargs:
            load_kwargs["device_map"] = "mps"
        
        # Aumentamos la precisión para compensar la falta de cuantización
        load_kwargs["torch_dtype"] = torch.float16
    else:
        # En otras plataformas (Windows/Linux), mantenemos la configuración original
        # que puede incluir bitsandbytes si está activado
        logger.info("Usando configuración estándar para plataforma no-Mac")
        
        # Verificar si se ha habilitado 8bit o 4bit
        use_8bit = os.getenv('USE_8BIT', 'False').lower() == 'true'
        use_4bit = os.getenv('USE_4BIT', 'False').lower() == 'true'
        
        # Aplicar configuración de bitsandbytes si está activada
        if use_8bit:
            logger.info("Activando cuantización de 8 bits con bitsandbytes")
            load_kwargs["load_in_8bit"] = True
        elif use_4bit:
            logger.info("Activando cuantización de 4 bits con bitsandbytes")
            load_kwargs["load_in_4bit"] = True
    
    try:
        logger.info(f"Intentando cargar modelo con PyTorch: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        logger.info(f"Modelo cargado exitosamente con PyTorch: {model_path}")
        return model, tokenizer, model_path
    except Exception as e:
        error_str = str(e)
        if "Access to model" in error_str and "is restricted" in error_str:
            logger.warning(f"ACCESO RESTRINGIDO: El modelo {model_path} requiere autenticación en Hugging Face.")
            logger.warning("Para usar este modelo, debes ejecutar 'huggingface-cli login' e ingresar tu token.")
            logger.warning(f"Cambiando automáticamente al modelo abierto: {fallback_model}")
            
            # Cargar modelo alternativo
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModelForCausalLM.from_pretrained(fallback_model, **load_kwargs)
            return model, tokenizer, fallback_model
        else:
            # Si es otro tipo de error, reenviar la excepción
            logger.error(f"Error al cargar el modelo: {error_str}")
            raise

# Clase base para modelos
class BaseModel:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

# Clase para usar la API de Hugging Face
class APIModel(BaseModel):
    def __init__(self, model_name: str, api_token: str, timeout: int = 30):
        """
        Inicializa el cliente de la API de Hugging Face.
        
        Args:
            model_name (str): Nombre del modelo en Hugging Face
            api_token (str): Token de la API de Hugging Face
            timeout (int): Timeout para las llamadas a la API
        """
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        self.timeout = timeout
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Genera texto usando la API de Hugging Face.
        
        Args:
            prompt (str): Texto de entrada
            kwargs: Argumentos adicionales para la generación
            
        Returns:
            str: Texto generado
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_length", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                return generated_text.strip()
            else:
                raise ValueError(f"Respuesta inesperada de la API: {result}")
                
        except Exception as e:
            logger.error(f"Error al generar texto con la API de {self.model_name}: {str(e)}")
            raise

# Clase para modelo local
class LocalModel(BaseModel):
    def __init__(self, model, tokenizer, device):
        """
        Inicializa el modelo local.
        
        Args:
            model: Modelo de transformers
            tokenizer: Tokenizer del modelo
            device: Dispositivo para inferencia
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Genera texto usando el modelo local.
        
        Args:
            prompt (str): Texto de entrada
            kwargs: Argumentos adicionales para la generación
            
        Returns:
            str: Texto generado
        """
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + kwargs.get("max_length", 512),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 50),
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
        except Exception as e:
            logger.error(f"Error al generar texto localmente: {str(e)}")
            raise

class RAGSystem:
    def __init__(
        self,
        model_path: str = os.getenv('BASE_MODEL_PATH', 'models/finetuned_model'),
        embeddings_dir: str = os.getenv('EMBEDDINGS_DIR', 'data/embeddings'),
        device: str = os.getenv('DEVICE', 'mps')
    ):
        """
        Inicializa el sistema RAG con la siguiente lógica de prioridad:
        1. Si USE_API=False:
           - Intenta cargar modelo fine-tuneado local
           - Si falla por memoria o no existe, usa API como fallback
        2. Si USE_API=True:
           - Usa directamente la API
        """
        self.device = device
        self.embeddings_dir = Path(embeddings_dir)
        
        # Obtener configuración
        self.use_api = os.getenv('USE_API', 'True').lower() == 'true'
        self.api_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
        self.base_model = os.getenv('BASE_MODEL_NAME')
        self.fallback_model = os.getenv('FALLBACK_MODEL_NAME')
        
        # Verificar si existe modelo fine-tuneado
        model_exists = os.path.exists(model_path) and os.path.isdir(model_path)
        
        if not self.use_api and model_exists:
            # Intentar usar modelo local primero
            try:
                available_memory = get_available_memory_gb()
                if available_memory < 16:
                    raise MemoryError(f"Memoria insuficiente ({available_memory:.2f} GB) para cargar modelo local")
                
                logger.info(f"Intentando cargar modelo fine-tuneado local desde: {model_path}")
                self.model = self._load_local_model(model_path)
                logger.info("Modelo local cargado exitosamente")
                return
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo local: {str(e)}")
                logger.info("Cambiando a API como fallback")
        
        # Si llegamos aquí, usamos la API (ya sea por configuración o como fallback)
        try:
            logger.info("Inicializando modelo via API")
            self.model = self._initialize_api_model()
        except Exception as e:
            raise RuntimeError(f"No se pudo inicializar ningún modelo (local ni API): {str(e)}")
        
        # Cargar modelo de embeddings
        embedding_model_name = 'hiiamsid/sentence_similarity_spanish_es'  # Modelo fijo para coincidir con el índice
        try:
            logger.info(f"Intentando cargar modelo de embeddings: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Modelo de embeddings cargado: {embedding_model_name}")
            logger.info(f"Dimensión del modelo: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.warning(f"Error al cargar modelo de embeddings principal: {str(e)}")
            logger.info("Usando modelo de respaldo con la misma dimensionalidad")
            self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            logger.info(f"Dimensión del modelo de respaldo: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Inicializar el almacén vectorial
        self.vector_store = self._initialize_vector_store()

        # Configurar umbral de similitud
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.1'))  # Umbral más permisivo
        
    def _load_local_model(self, model_path: str):
        """Carga el modelo local con la configuración apropiada"""
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return LocalModel(model, tokenizer, self.device)

    def _initialize_api_model(self) -> APIModel:
        """Inicializa el modelo de API con fallback"""
        if not self.api_token:
            raise ValueError("Se requiere HUGGING_FACE_HUB_TOKEN para usar la API")
            
        try:
            # Intentar primero con Mistral
            model = APIModel(self.base_model, self.api_token)
            # Verificar que funciona
            model.generate("Test", max_length=10)
            logger.info(f"Usando modelo principal via API: {self.base_model}")
            return model
        except Exception as e:
            logger.warning(f"Error con modelo principal: {str(e)}")
            logger.info(f"Intentando con modelo de fallback: {self.fallback_model}")
            
            try:
                # Fallback a TinyLlama
                model = APIModel(self.fallback_model, self.api_token)
                # Verificar que funciona
                model.generate("Test", max_length=10)
                logger.info(f"Usando modelo de fallback via API: {self.fallback_model}")
                return model
            except Exception as e:
                raise RuntimeError(f"No se pudo inicializar ningún modelo: {str(e)}")

    def _initialize_vector_store(self) -> VectorStore:
        """
        Inicializa el almacén vectorial adecuado según el entorno.
        
        Returns:
            VectorStore: Implementación del almacén vectorial
        """
        if ENVIRONMENT == 'production':
            # Usar Pinecone en producción
            try:
                return PineconeVectorStore(PINECONE_INDEX_NAME)
            except (ValueError, NameError) as e:
                logger.error(f"Error al inicializar Pinecone: {str(e)}")
                logger.warning("Fallback a FAISS local")
                # Fallback a FAISS si hay error
        
        # Usar FAISS en desarrollo o como fallback
        index_path = str(self.embeddings_dir / 'faiss_index.bin')
        metadata_path = str(self.embeddings_dir / 'metadata.csv')
        return FAISSVectorStore(index_path, metadata_path)
        
    def _expand_query(self, query: str) -> str:
        """
        Expande la consulta para mejorar la búsqueda semántica.
        """
        query_lower = query.lower()
        expanded_query = query
        
        # Palabras clave y sus expansiones
        keywords = {
            'sanción': ['sanción', 'sanciones', 'castigo', 'penalidad', 'disciplina', 'régimen disciplinario'],
            'agredir': ['agredir', 'agresión', 'violencia', 'ataque', 'golpear'],
            'profesor': ['profesor', 'docente', 'maestro', 'autoridad universitaria'],
            'alumno': ['alumno', 'estudiante', 'cursante'],
            'suspensión': ['suspensión', 'expulsión', 'separación'],
            'físicamente': ['físicamente', 'físico', 'corporal', 'material']
        }
        
        # Buscar palabras clave en la consulta
        for key, expansions in keywords.items():
            if any(word in query_lower for word in expansions):
                expanded_query = f"{expanded_query} {' '.join(expansions)}"
        
        # Agregar referencias a artículos relevantes
        if any(word in query_lower for word in keywords['sanción'] + keywords['agredir']):
            expanded_query = f"{expanded_query} artículo 13 artículo 14 artículo 15 régimen disciplinario"
        
        logger.info(f"Consulta expandida: {expanded_query}")
        return expanded_query
        
    def retrieve_relevant_chunks(self, query: str, k: int = None) -> List[Dict]:
        """
        Recupera chunks relevantes para una consulta.
        
        Args:
            query (str): Consulta del usuario
            k (int): Número de chunks a recuperar
            
        Returns:
            List[Dict]: Lista de chunks relevantes con metadatos
        """
        if k is None:
            k = int(os.getenv('RAG_NUM_CHUNKS', 5))
            
        # Preprocesar la consulta
        query = query.strip()
        if not query:
            return []
            
        # Generar embedding de la consulta
        try:
            # Expandir la consulta para mejorar la búsqueda
            expanded_query = self._expand_query(query)
            logger.info(f"Consulta expandida: {expanded_query}")
            
            query_embedding = self.embedding_model.encode(
                [expanded_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            logger.info(f"Embedding generado con forma: {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Error al generar embedding de consulta: {str(e)}", exc_info=True)
            return []
            
        # Realizar búsqueda
        try:
            # Verificar que el vector_store está inicializado
            if not hasattr(self, 'vector_store'):
                logger.error("vector_store no está inicializado")
                return []
                
            logger.info("Iniciando búsqueda en vector_store")
            results = self.vector_store.search(query_embedding, k=k*2)
            logger.info(f"Búsqueda completada, resultados obtenidos: {len(results)}")
            
            # Verificar si los resultados son relevantes
            if not results:
                logger.warning("No se encontraron chunks relevantes para la consulta.")
                
                # Verificar si la consulta es sobre trámites comunes
                query_lower = query.lower()
                tramites_keywords = {
                    'constancia': 'a) Constancia de Alumno Regular',
                    'alumno regular': 'a) Constancia de Alumno Regular',
                    'baja': 'b) Baja de Materias',
                    'dar de baja': 'b) Baja de Materias',
                    'reincorporación': 'c) Reincorporación',
                    'reincorporacion': 'c) Reincorporación',
                    'recursada': 'd) Recursadas',
                    'recursar': 'd) Recursadas'
                }
                
                # Buscar coincidencias en las palabras clave
                tramite_encontrado = None
                for keyword, tramite in tramites_keywords.items():
                    if keyword in query_lower:
                        tramite_encontrado = tramite
                    break
            
                if tramite_encontrado:
                    # Usar la información base del trámite correspondiente
                    if 'constancia' in query_lower or 'alumno regular' in query_lower:
                        response = """Para tramitar la constancia de alumno regular:
1. **Tramitar en el Sitio de Inscripciones**
2. Ingresar con DNI y contraseña
3. Seleccionar "Constancia de alumno regular"
4. Imprimir y presentar en ventanilla del Ciclo Biomédico con Libreta/DNI"""
                    elif 'baja' in query_lower:
                        response = """Para dar de baja una materia:
- **Plazo máximo**: 2 semanas antes del primer parcial o hasta el 25% de la cursada
- **Pasos**:
  1. Tramitar en Sitio de Inscripciones
  2. Seleccionar "Baja de asignatura"
  3. No requiere presentación en ventanilla si el estado es "Resuelto Positivamente" """
                    elif 'reincorporación' in query_lower or 'reincorporacion' in query_lower:
                        response = """Para solicitar la reincorporación:
- **Primera reincorporación**: 
  - Trámite automático en el sistema
  - No requiere presentación en ventanilla
- **Segunda reincorporación o más**:
  1. Tramitar en Sitio de Inscripciones
  2. Presentar documentación en ventanilla
  3. La Comisión de Readmisión evaluará el caso"""
                    else:  # recursada
                        response = """Para solicitar una recursada:
- **Si figura como BAJA en cursada anterior**:
  - Sin arancel
  - Inscripción normal como primera vez
- **Si NO figura como BAJA**:
  1. Generar trámite
  2. Pagar arancel en Tesorería
  3. Presentar comprobante en buzones del Ciclo Biomédico"""
                    
                    return {
                        "query": query,
                        "response": response,
                        "relevant_chunks": [],
                        "sources": ["Información de Trámites Comunes"]
                    }
                else:
                    # Si no es un trámite común, usar respuesta estándar con derivación por email
                    has_greeting = any(word in query.lower() for word in greeting_words)
                    if has_greeting:
                        standard_no_info_response = f"👨‍⚕️ ¡Hola! Soy DrCecim. Lo siento, no tengo información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas. Si tienes otras preguntas sobre temas relacionados con la Facultad de Medicina, no dudes en consultarme."
                    else:
                        standard_no_info_response = f"👨‍⚕️ Lo siento, no tengo información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas. Si tienes otras preguntas sobre temas relacionados con la Facultad de Medicina, no dudes en consultarme."
                    
                return {
                    "query": query,
                        "response": standard_no_info_response,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Filtrar por similitud y duplicados
            filtered_results = []
            seen_content = set()
            
            for result in results:
                text = result.get('text', '').strip()
                # Crear una versión simplificada del texto para comparación
                simple_text = ' '.join(text.lower().split())
                
                # Verificar similitud usando el campo correcto
                similarity = result.get('similarity', 0.0)
                filename = result.get('filename', '')
                
                logger.info(f"Procesando resultado - Similitud: {similarity}, Archivo: {filename}")
                
                # Dar prioridad a chunks del régimen disciplinario si la consulta es sobre sanciones
                if any(word in query.lower() for word in ['sanción', 'sanciones', 'agredir', 'agresión']):
                    if "Regimen_Disciplinario.pdf" in filename:
                        # Reducir el umbral para documentos relevantes
                        if similarity >= (self.similarity_threshold * 0.5):  # Umbral más permisivo para documentos relevantes
                            if simple_text not in seen_content:
                                seen_content.add(simple_text)
                                filtered_results.append(result)
                                logger.info(f"Chunk de Régimen Disciplinario aceptado con similitud: {similarity:.3f}")
            else:
                        # Mantener umbral normal para otros documentos
                        if similarity >= self.similarity_threshold and simple_text not in seen_content:
                            seen_content.add(simple_text)
                            filtered_results.append(result)
                            logger.info(f"Chunk de otro documento aceptado con similitud: {similarity:.3f}")
                        else:
                            # Para otras consultas, usar el umbral normal
                            if similarity >= self.similarity_threshold and simple_text not in seen_content:
                                seen_content.add(simple_text)
                                filtered_results.append(result)
                                logger.info(f"Chunk aceptado con similitud: {similarity:.3f}")
            
            # Ordenar por similitud y limitar a k resultados
            filtered_results = sorted(filtered_results, key=lambda x: x.get('similarity', 0.0), reverse=True)[:k]
            
            # Logging de resultados
            logger.info(f"Recuperados {len(filtered_results)} chunks únicos de {len(results)} totales")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error en la búsqueda de chunks: {str(e)}", exc_info=True)
            return []
        
    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        """
        Genera una respuesta basada en el contexto y la consulta.
        
        Args:
            query (str): Consulta del usuario
            context (str): Contexto relevante recuperado
            sources (List[str]): Lista de fuentes de información
            
        Returns:
            str: Respuesta generada
        """
        # Prompt base para el asistente administrativo
        system_prompt = """Eres un asistente administrativo especializado de la Universidad de Buenos Aires (UBA).
Tu rol es proporcionar información precisa sobre trámites y procedimientos administrativos.

INSTRUCCIONES IMPORTANTES:
1. Responde SOLO con información verificada que encuentres en el contexto proporcionado
2. Si no tienes información suficiente en el contexto, sigue estas pautas:

A) Para trámites comunes conocidos, proporciona esta información base:

PREGUNTAS FRECUENTES Y RESPUESTAS ESTÁNDAR:

1. CONSTANCIAS Y CERTIFICADOS:
   - "¿Dónde puedo tramitar la constancia de alumno regular?"
   Respuesta base:
   - **Paso 1:** Ingresar DNI y contraseña en Sitio de Inscripciones
   - **Paso 2:** Seleccionar "Constancia de alumno regular"
   - **Paso 3:** Imprimir formulario (1 hoja con 3 certificados)
   - **Paso 4:** Presentar en ventanilla del Ciclo Biomédico con Libreta Universitaria o DNI

2. BAJAS Y MODIFICACIONES:
   a) "¿Cuánto tiempo tengo para dar de baja una materia?"
   Respuesta base:
   - Plazo máximo: **2 semanas antes del primer parcial**
   - O **hasta el 25% de la cursada** en materias sin parcial
   - **Paso 1:** Ingresar al Sitio de Inscripciones
   - **Paso 2:** Seleccionar "Baja de asignatura"
   - **Paso 3:** Si aparece "Resuelto Positivamente", no requiere más trámites

   b) "¿Cómo anulo una inscripción a final?"
   Respuesta base:
   - Acudir a ventanilla del Ciclo Biomédico
   - Presentar número de constancia del trámite de inscripción
   
   c) "¿Qué hago si no logro inscribirme o no salgo asignado?"
   Respuesta base:
   - Dirigirse a la cátedra o departamento correspondiente
   - Solicitar inclusión en lista
   - Presentar Libreta Universitaria o DNI

3. REINCORPORACIONES:
   - "¿Cómo solicito la reincorporación a la Carrera?"
   Respuesta base:
   - **Primera reincorporación:**
     * Trámite automático en sistema
     * No requiere presentación en ventanilla
   - **Segunda reincorporación o más:**
     * Realizar trámite en Sitio de Inscripciones
     * Imprimir documentación (2 hojas: certificado y constancia)
     * Presentar en ventanilla del Ciclo Biomédico
     * Evaluación por Comisión de Readmisión

4. RECURSADAS Y NUEVAS CURSADAS:
   a) "¿Cómo solicito una recursada?"
   Respuesta base:
   - **Si figura como BAJA:**
     * Generar trámite
     * Inscripción normal sin arancel
   - **Si NO figura como BAJA:**
     * Generar trámite
     * Imprimir talón de pago
     * Pagar en Tesorería
     * Presentar comprobante en buzones

   b) "¿Cómo solicito una tercera cursada?"
   Respuesta base:
   - **Paso 1:** Ingresar al Sitio de Inscripciones
   - **Paso 2:** Seleccionar "3º Cursada"
   - **Paso 3:** Imprimir constancia y certificado
   - **Si figura BAJA en cursadas anteriores:**
     * Inscripción normal sin arancel
   - **Si NO figura BAJA:**
     * Pagar arancel en Tesorería
     * Presentar comprobante en buzón

   c) "¿Cómo solicito una cuarta cursada?"
   Respuesta base:
   - **Paso 1:** Ingresar al Sitio de Inscripciones
   - **Paso 2:** Seleccionar "4º Cursada o más"
   - **Paso 3:** Imprimir documentación
   - **Paso 4:** Presentar en ventanilla con Libreta
   - **Paso 5:** Acudir a Dirección de Alumnos

5. PRÓRROGAS Y EXTENSIONES:
   - "¿Cómo hago el trámite de prórroga de materias?"
   Respuesta base:
   - **Primera o segunda prórroga:**
     * Tramitar en Sitio de Inscripciones
     * Seleccionar "Prórroga de asignatura"
     * Trámite automático resuelto positivamente
   - **Tercera prórroga o superior:**
     * Realizar trámite
     * Imprimir constancia
     * Presentar en ventanilla con Libreta Universitaria

B) Para consultas sin información disponible:
   - Indica claramente que no tienes la información específica
   - Sugiere contactar a alumnos@fmed.uba.ar para obtener información precisa
   - Mantén un tono amable y profesional al derivar la consulta

3. Mantén un tono profesional pero amigable
4. Estructura las respuestas en pasos claros cuando sea apropiado
5. Incluye detalles específicos sobre documentación requerida
6. Menciona dónde debe realizarse cada trámite

FORMATO DE RESPUESTA:
- Usa viñetas o números para listar pasos
- Destaca información importante en **negrita**
- Separa secciones con líneas si es necesario
- Incluye advertencias o notas importantes cuando sea relevante

Contexto proporcionado:
{context}

Consulta del usuario: {query}

Respuesta:"""

        # Preparar el prompt completo
        prompt = system_prompt.format(
            context=context,
            query=query
        )

        try:
            # Generar respuesta
            response = self.model.generate(prompt)
            
            # Si no hay información suficiente en el contexto
            if "no tengo información suficiente" in response.lower():
                return "Lo siento, no tengo información específica sobre ese trámite en este momento. Te sugiero consultar directamente en la ventanilla del Ciclo Biomédico o en la Dirección de Alumnos para obtener la información más actualizada."
            
            # Agregar fuentes si están disponibles
            if sources:
                response += "\n\nFuente(s): " + ", ".join(sources)
            
            return response
            
        except Exception as e:
            logger.error(f"Error al generar respuesta: {str(e)}")
            return "Lo siento, hubo un error al procesar tu consulta. Por favor, intenta nuevamente o consulta directamente en la Dirección de Alumnos."
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta utilizando RAG para generar una respuesta.
        """
        try:
            # Establecer el número de chunks por defecto
            num_chunks = int(os.getenv('RAG_NUM_CHUNKS', 3))
            logger.info(f"Procesando consulta: {query}")
            
            # Encontrar fragmentos relevantes
            relevant_chunks = self.retrieve_relevant_chunks(query, k=num_chunks)
            
            # Verificar si se encontraron chunks relevantes
            if not relevant_chunks:
                logger.warning("No se encontraron chunks relevantes para la consulta.")
                
                # Verificar si la consulta es sobre sanciones o agresiones
                query_lower = query.lower()
                if any(word in query_lower for word in ['sanción', 'sanciones', 'agredir', 'agresión']):
                    # Intentar una nueva búsqueda con umbral más bajo para el Régimen Disciplinario
                    logger.info("Intentando búsqueda específica en Régimen Disciplinario...")
                    self.similarity_threshold = 0.1  # Reducir temporalmente el umbral
                    relevant_chunks = self.retrieve_relevant_chunks(query, k=num_chunks)
                    self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))  # Restaurar umbral original
                
                if not relevant_chunks:
                    standard_no_info_response = f"👨‍⚕️ Lo siento, no encontré información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas."
                
                return {
                    "query": query,
                    "response": standard_no_info_response,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Construir contexto
            context_chunks = []
            sources = []
            
            for chunk in relevant_chunks:
                if "content" in chunk and chunk["content"].strip():
                    content = chunk["content"]
                elif "text" in chunk and chunk["text"].strip():
                    content = chunk["text"]
                else:
                    continue
                
                source = ""
                if "filename" in chunk and chunk["filename"]:
                    source = os.path.basename(chunk["filename"]).replace('.pdf', '')
                    if source and source not in sources:
                        sources.append(source)
                
                formatted_chunk = f"Información de {source}:\n{content}"
                context_chunks.append(formatted_chunk)
                logger.info(f"Agregado chunk relevante de {source}")
            
            # Unir los chunks para formar el contexto
            context = '\n\n'.join(context_chunks)
            
            if not context.strip():
                logger.warning("No se encontró contexto suficientemente relevante")
                standard_no_info_response = f"👨‍⚕️ Lo siento, no encontré información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas."
                
                return {
                    "query": query,
                    "response": standard_no_info_response,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            logger.info(f"Se encontraron {len(context_chunks)} fragmentos relevantes de {len(sources)} fuentes")
            
            # Generar respuesta
            response = self.generate_response(query, context, sources)
            
            # Agregar fuentes al final de la respuesta
            if sources:
                clean_sources = [source.replace('_', ' ').replace('-', ' ') for source in sources]
                sources_text = ", ".join(clean_sources)
                response = f"{response}\n\nEsta información la puedes encontrar en: {sources_text}"
            
            return {
                "query": query,
                "response": response,
                "relevant_chunks": relevant_chunks,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error en process_query: {str(e)}", exc_info=True)
            error_response = f"👨‍⚕️ Lo siento, tuve un problema procesando tu consulta. Por favor, intenta de nuevo."
            return {
                "query": query,
                "response": error_response,
                "error": str(e)
            }

def main():
    """Función principal para ejecutar el sistema RAG."""
    # Inicializar sistema RAG
    rag = RAGSystem()
    
    # Ejemplo de uso
    while True:
        query = input("\nIngrese su consulta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
            
        try:
            result = rag.process_query(query)
            print("\nRespuesta:", result['response'])
            print("\nFuentes utilizadas:")
            for chunk in result['relevant_chunks']:
                if 'filename' in chunk and 'chunk_index' in chunk:
                    print(f"- {chunk['filename']} (chunk {chunk['chunk_index']})")
                else:
                    print(f"- {chunk.get('id', 'unknown')}")
        except Exception as e:
            logger.error(f"Error al procesar la consulta: {str(e)}")
            print("Lo siento, hubo un error al procesar tu consulta.")

if __name__ == "__main__":
    main() 