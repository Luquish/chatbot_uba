import os
import logging
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
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
import random
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity

# Crear directorio de logs si no existe
Path("logs").mkdir(exist_ok=True)

# Nueva estructura de intenciones para clasificación semántica
INTENT_EXAMPLES = {
    'saludo': {
        'examples': [
            "hola",
            "buenos días",
            "qué tal",
            "buenas"
        ],
        'context': "El usuario está iniciando la conversación o saludando"
    },
    'pregunta_nombre': {
        'examples': [
            "como me llamo yo",
            "cual es mi nombre",
            "sabes como me llamo",
            "como sabes mi nombre",
            "por que sabes mi nombre",
            "de donde sacaste mi nombre",
            "como conseguiste mi nombre",
            "por que conoces mi nombre"
        ],
        'context': "El usuario pregunta específicamente sobre cómo conocemos su nombre"
    },
    'cortesia': {
        'examples': [
            "cómo estás",
            "como estas",
            "como te sentis",
            "como va",
            "todo bien"
        ],
        'context': "El usuario hace una pregunta de cortesía"
    },
    'referencia_anterior': {
        'examples': [
            "resúmeme eso",
            "resumeme eso",
            "puedes resumir lo anterior",
            "podrias resumirme el mensaje anterior",
            "podrias resumirme ese texto",
            "resume el mensaje anterior",
            "explica de nuevo",
            "explícame eso",
            "explicame eso de nuevo",
            "acorta esa explicación",
            "simplifica lo que dijiste",
            "dímelo más corto",
            "dimelo mas corto",
            "puedes abreviar",
            "puedes hacer un resumen"
        ],
        'context': "El usuario está pidiendo un resumen o clarificación del mensaje anterior"
    },
    'pregunta_capacidades': {
        'examples': [
            "qué podés hacer",
            "en qué me podés ayudar",
            "para qué servís",
            "qué tipo de consultas puedo hacer",
            "que sabes",
            "que sabes hacer",
            "que podes hacer",
            "que me podes decir",
            "cuales son tus funciones",
            "que funciones tenes",
            "que te puedo preguntar",
            "que puedo preguntarte",
            "que tipos de dudas puedo consultar",
            "en que me podes ayudar"
        ],
        'context': "El usuario quiere saber las capacidades del bot"
    },
    'identidad': {
        'examples': [
            "quién sos",
            "cómo te llamás",
            "sos un bot",
            "sos una persona"
        ],
        'context': "El usuario pregunta sobre la identidad del bot"
    },
    'consulta_administrativa': {
        'examples': [
            "cómo hago un trámite",
            "necesito una constancia",
            "dónde presento documentación",
            "quiero dar de baja una materia",
            "cuántas materias debo aprobar",
            "en cuánto tiempo tengo que terminar la carrera",
            "cómo se define el año académico",
            "qué derechos tengo para inscribirme"
        ],
        'context': "El usuario necesita información sobre trámites administrativos o condiciones de regularidad"
    },
    'consulta_academica': {
        'examples': [
            "cuándo es el parcial",
            "dónde encuentro el programa",
            "cómo es la cursada",
            "qué necesito para aprobar",
            "cómo se evalúa la calidad de enseñanza",
            "qué materias puedo cursar"
        ],
        'context': "El usuario necesita información académica"
    },
    'consulta_medica': {
        'examples': [
            "me duele la cabeza",
            "tengo síntomas de",
            "dónde puedo consultar por un dolor",
            "necesito un diagnóstico",
            "tengo fiebre"
        ],
        'context': "El usuario hace una consulta médica que no podemos responder"
    },
    'consulta_reglamento': {
        'examples': [
            "qué dice el reglamento sobre",
            "está permitido",
            "cuáles son las normas",
            "qué pasa si no cumplo",
            "qué medidas toman si me porto mal",
            "quién decide las sanciones",
            "qué castigos hay",
            "para qué sirven las medidas disciplinarias",
            "qué sanciones aplican",
            "si cometo una falta",
            "quién evalúa mi comportamiento",
            "qué pasa si rompo las reglas"
        ],
        'context': "El usuario pregunta sobre normativas, reglamentos y medidas disciplinarias"
    },
    'agradecimiento': {
        'examples': [
            "perfecto",
            "gracias",
            "ok",
            "okk",
            "okey",
            "okay",
            "dale",
            "listo",
            "entendido",
            "genial",
            "excelente",
            "bárbaro",
            "buenísimo",
            "joya"
        ],
        'context': "El usuario agradece o confirma que entendió la información"
    }
}

GREETING_WORDS = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches', 'buen dia', 'saludos', 'que tal']

# Lista de emojis para enriquecer las respuestas
information_emojis = ["📚", "📖", "ℹ️", "📊", "🔍", "📝", "📋", "📈", "📌", "🧠"]
greeting_emojis = ["👋", "😊", "🤓", "👨‍⚕️", "👩‍⚕️", "🎓", "🌟"]
warning_emojis = ["⚠️", "❗", "⚡", "🚨"]
success_emojis = ["✅", "💫", "🎉", "💡"]
medical_emojis = ["🏥", "👨‍⚕️", "👩‍⚕️", "🩺"]

# Palabras clave para expansión de consultas
QUERY_EXPANSIONS = {
    'inscripcion': ['inscribir', 'anotarse', 'anotar', 'registrar', 'inscripto'],
    'constancia': ['certificado', 'comprobante', 'papel', 'documento'],
    'regular': ['regularidad', 'condición', 'estado', 'situación'],
    'final': ['examen', 'evaluación', 'rendir', 'dar'],
    'recursada': ['recursar', 'volver a cursar', 'segunda vez'],
    'correlativa': ['correlatividad', 'requisito', 'necesito', 'puedo cursar'],
    'baja': ['dar de baja', 'abandonar', 'dejar', 'salir'],
}

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Salida a consola
        logging.FileHandler(Path('logs') / 'app.log')  # Salida a archivo
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
        
        # Tendríamos:
        self.conversation_histories = {}  # Diccionario: user_id -> historial
        self.max_history_length = 5
        
        # Obtener configuración
        self.use_api = os.getenv('USE_API', 'True').lower() == 'true'
        self.api_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
        self.base_model = os.getenv('BASE_MODEL_NAME')
        self.fallback_model = os.getenv('FALLBACK_MODEL_NAME')
        
        # Normalizar los ejemplos de intenciones para mejorar la detección
        self.normalized_intent_examples = self._normalize_intent_examples()
        logger.info("Ejemplos de intenciones normalizados para mejorar la clasificación")
        
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
        embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'hiiamsid/sentence_similarity_spanish_es')
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
                return []
            
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
        
    def _format_source_name(self, source: str) -> str:
        """
        Formatea el nombre de la fuente eliminando extensiones y caracteres especiales.
        
        Args:
            source (str): Nombre original de la fuente (ej: "Condiciones_Regularidad.pdf")
            
        Returns:
            str: Nombre formateado (ej: "Condiciones de Regularidad")
        """
        # Eliminar extensión .pdf
        source = source.replace('.pdf', '')
        
        # Reemplazar guiones bajos y guiones medios por espacios
        source = source.replace('_', ' ').replace('-', ' ')
        
        # Capitalizar palabras
        source = ' '.join(word.capitalize() for word in source.split())
        
        return source

    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        """
        Genera una respuesta usando el modelo de lenguaje
        """
        # Determinar el emoji según la intención
        intent, _ = self._get_query_intent(query)
        emoji = random.choice({
            'saludo': greeting_emojis,
            'pregunta_capacidades': information_emojis,
            'consulta_administrativa': information_emojis,
            'consulta_academica': information_emojis,
            'consulta_medica': medical_emojis,
            'consulta_reglamento': warning_emojis
        }.get(intent, information_emojis))
        
        # Construir el prompt con el contexto de la intención y las FAQs
        intent_context = INTENT_EXAMPLES.get(intent, {}).get('context', 'Consulta general')
        
        faqs = """
[PREGUNTAS FRECUENTES]
1. Constancia de alumno regular:
   Puedes tramitar la constancia de alumno regular en el Sitio de Inscripciones siguiendo estos pasos:
   - Paso 1: Ingresar tu DNI y contraseña.
   - Paso 2: Seleccionar la opción "Constancia de alumno regular" en el inicio de trámites.
   - Paso 3: Imprimir la constancia. Luego, deberás presentarte con tu Libreta Universitaria o DNI y el formulario impreso (1 hoja que posee 3 certificados de alumno regular) en la ventanilla del Ciclo Biomédico.

2. Baja de materia:
   El tiempo máximo para dar de baja una materia es:
   - 2 semanas antes del primer parcial, o
   - Hasta el 25% de la cursada en asignaturas sin examen parcial.
   Para dar de baja una materia, sigue estos pasos en el Sitio de Inscripciones:
   - Paso 1: Ingresar tu DNI y contraseña.
   - Paso 2: Seleccionar "Baja de asignatura".
   - Paso 3: Imprimir el certificado de baja. Una vez finalizado el trámite, el estado será "Resuelto Positivamente" y no deberás acudir a la Dirección de Alumnos.

3. Anulación de inscripción a final:
   Para anular la inscripción a un final, debes acudir a la ventanilla del Ciclo Biomédico presentando el número de constancia generado durante el trámite de inscripción.

4. No lograr inscripción o asignación a materia:
   Si no logras inscribirte o no te asignan una materia, debes dirigirte a la cátedra o departamento correspondiente y solicitar la inclusión en lista, presentando tu Libreta Universitaria o DNI.

5. Reincorporación:
   La reincorporación se solicita a través del Sitio de Inscripciones, seleccionando la opción "Reincorporación a la carrera":
   - Para la 1ª reincorporación: El trámite es automático y aparece resuelto positivamente en el sistema, sin necesidad de trámite en ventanilla.
   - Si ya fuiste reincorporado anteriormente: Debes realizar el trámite, imprimirlo (consta de 2 hojas: 1 certificado y 1 constancia) y presentarlo en la ventanilla del Ciclo Biomédico, donde la Comisión de Readmisión resolverá tu caso.

6. Recursada (inscripción por segunda vez):
   Para solicitar una recursada, genera el trámite en el Sitio de Inscripciones siguiendo estos pasos:
   - Paso 1: Ingresar tu DNI y contraseña.
   - Paso 2: Seleccionar "Recursada".
   El trámite es automático y, si aparece resuelto positivamente en el sistema, no necesitas acudir a ventanilla.
   - Si en el sistema apareces como dado DE BAJA en la cursada anterior, solo debes generar el trámite y te inscribirás como la primera vez, sin abonar arancel.
   - Si no apareces dado DE BAJA, deberás:
     1. Realizar el trámite.
     2. Generar e imprimir el talón de pago.
     3. Pagar en la Dirección de Tesorería.
     4. Presentar un comprobante de pago en los buzones del Ciclo Biomédico.

7. Tercera cursada:
   Para solicitar la tercera cursada, sigue estos pasos en el Sitio de Inscripciones:
   - Paso 1: Ingresar tu DNI y contraseña.
   - Paso 2: Seleccionar "3º Cursada".
   - Paso 3: Imprimir la constancia y el certificado.
   Luego:
   - Si figuras como dado DE BAJA en las dos cursadas anteriores, te inscribes como si fuera la primera vez sin abonar arancel.
   - Si no, debes:
     1. Realizar el trámite.
     2. Generar e imprimir el talón de pago.
     3. Pagar en la Dirección de Tesorería.
     4. Presentar un comprobante de pago en el buzón del Ciclo Biomédico.

8. Cuarta cursada o más:
   Para la cuarta cursada o más, genera el trámite en el Sitio de Inscripciones con los siguientes pasos:
   - Paso 1: Dirigirte a Inscripciones.
   - Paso 2: Ingresar tu DNI y contraseña.
   - Paso 3: Seleccionar "4º Cursada o más".
   - Paso 4: Imprimir la constancia y el certificado.
   Luego, deberás presentarte con tu Libreta Universitaria y las constancias impresas en la ventanilla del Ciclo Biomédico y acudir a la Dirección de Alumnos.

9. Prórroga de materias:
   Para solicitar la prórroga de una asignatura, sigue estos pasos en el Sitio de Inscripciones:
   - Paso 1: Dirigirte a Inscripciones.
   - Paso 2: Ingresar tu DNI y contraseña.
   - Paso 3: Seleccionar "Prórroga de asignatura".
   - Paso 4: Imprimir la constancia.
   Si se trata de la primera o segunda prórroga, el trámite se resuelve positivamente. Si es la tercera o una prórroga superior, deberás presentar la constancia impresa junto con tu Libreta Universitaria en la ventanilla del Ciclo Biomédico.
"""

        # Solo incluir las fuentes en el prompt si no son nulas y la lista no está vacía
        sources_text = ""
        if sources and len(sources) > 0:
            formatted_sources = [self._format_source_name(src) for src in sources]
            sources_text = f"\nFUENTES:\n{', '.join(formatted_sources)}"

        prompt = f"""[INST]
Como DrCecim, asistente virtual de la Facultad de Medicina UBA:

CONTEXTO DE LA CONSULTA:
{intent_context}

INFORMACIÓN RELEVANTE:
{context}

{faqs}

{sources_text}

CONSULTA:
{query}

INSTRUCCIONES:
1. Si la consulta coincide con alguna de las preguntas frecuentes, proporciona esa respuesta exacta sin mencionar fuentes
2. Si la consulta es similar pero no exacta a una pregunta frecuente, adapta la respuesta manteniendo la información precisa
3. Si la consulta requiere información de los documentos:
   - Integra la información naturalmente en la respuesta
   - Si es relevante mencionar la fuente, hazlo de forma natural en el contexto
   - Ejemplos de cómo mencionar fuentes:
     * "Según el Reglamento de Regularidad, ..."
     * "De acuerdo con las Condiciones de Regularidad, ..."
     * "Como establece el Régimen Disciplinario, ..."
4. Mantén el formato y la estructura de las respuestas frecuentes cuando corresponda
5. No hagas preguntas adicionales
6. Si es una consulta médica, deriva al profesional
7. No agregues una sección de "Fuentes:" al final del mensaje

[/INST]"""

        response = self.model.generate(prompt)
        return f"{emoji} {response}"

    def _handle_medical_query(self) -> str:
        """
        Genera una respuesta estándar para consultas médicas
        """
        responses = [
            f"{random.choice(medical_emojis)} Lo siento, no puedo responder consultas médicas. Por favor, consultá con un profesional de la salud o acercate a la guardia del Hospital de Clínicas.",
            f"{random.choice(medical_emojis)} Como asistente virtual, no estoy capacitado para responder consultas médicas. Te recomiendo consultar con un profesional médico o acudir al Hospital de Clínicas.",
            f"{random.choice(medical_emojis)} Disculpá, pero no puedo dar consejos médicos. Para este tipo de consultas, te sugiero:\n"
            "1. Consultar con un profesional médico\n"
            "2. Acudir a la guardia del Hospital de Clínicas\n"
            "3. En caso de emergencia, llamar al SAME (107)"
        ]
        return random.choice(responses)

    def _handle_outdated_info(self, response: str, source_date: str = None) -> str:
        """Manejo de información potencialmente desactualizada"""
        warning = "\n\n⚠️ Esta información corresponde al reglamento vigente. " \
                 "Para confirmar cualquier cambio reciente, consultá en alumnos@fmed.uba.ar"
        return f"{response}{warning}"

    def _normalize_intent_examples(self) -> Dict:
        """
        Normaliza los ejemplos de intenciones para hacer comparaciones más robustas
        """
        normalized_examples = {}
        
        for intent, data in INTENT_EXAMPLES.items():
            examples = data['examples']
            norm_examples = []
            
            for example in examples:
                # Aplicar la misma normalización que a las consultas
                norm_example = example.lower().strip()
                norm_example = unidecode(norm_example)  # Eliminar tildes
                norm_example = re.sub(r'[^\w\s]', '', norm_example)  # Eliminar signos de puntuación
                norm_example = re.sub(r'\s+', ' ', norm_example).strip()  # Normalizar espacios
                norm_examples.append(norm_example)
                
            normalized_examples[intent] = {
                'examples': norm_examples,
                'context': data['context']
            }
            
        return normalized_examples

    def _get_query_intent(self, query: str) -> Tuple[str, float]:
        """
        Determina la intención de la consulta usando similitud semántica
        """
        # Normalización del texto
        query_original = query
        query = query.lower().strip()
        query = unidecode(query)  # Eliminar tildes
        query = re.sub(r'[^\w\s]', '', query)  # Eliminar signos de puntuación
        query = re.sub(r'\s+', ' ', query).strip()  # Normalizar espacios
        
        if query_original != query:
            logger.info(f"Consulta normalizada: '{query_original}' → '{query}'")
            
        query_embedding = self.embedding_model.encode([query])[0]
        
        max_similarity = -1
        best_intent = 'desconocido'
        
        # Usar ejemplos normalizados
        for intent, data in self.normalized_intent_examples.items():
            examples = data['examples']
            example_embeddings = self.embedding_model.encode(examples)
            similarities = cosine_similarity([query_embedding], example_embeddings)[0]
            avg_similarity = np.mean(similarities)
            
            # Para debugging
            logger.debug(f"Intención: {intent}, similitud: {avg_similarity:.2f}")
            
            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                best_intent = intent
        
        return best_intent, max_similarity

    def _generate_conversational_response(self, query: str, intent: str, user_name: str = None) -> str:
        """
        Genera una respuesta conversacional basada en la intención detectada
        """
        context = INTENT_EXAMPLES[intent]['context'] if intent in INTENT_EXAMPLES else "Consulta general"
        
        # Determinar si es el primer mensaje o un saludo del usuario
        is_greeting = intent == 'saludo'
        is_courtesy = intent == 'cortesia'
        is_acknowledgment = intent == 'agradecimiento'
        is_capabilities = intent == 'pregunta_capacidades'
        
        # Lista de respuestas alegres para agradecimientos
        happy_responses = [
            "¡Me alegro de haber podido ayudarte! 😊",
            "¡Qué bueno que te sirvió la información! 🌟",
            "¡Genial! Estoy aquí para lo que necesites 💫",
            "¡Excelente! No dudes en consultarme cualquier otra duda 🎓",
            "¡Me pone contento poder ayudarte! 😊",
            "¡Perfecto! Seguimos en contacto 👋",
            "¡Bárbaro! Cualquier otra consulta, aquí estoy 🤓"
        ]
        
        # Lista de respuestas para preguntas sobre capacidades
        capabilities_responses = [
            "Soy un asistente especializado en:\n- Trámites administrativos de la facultad\n- Consultas sobre el reglamento y normativas\n- Información académica general\n- Procesos de inscripción y regularidad",
            "Puedo ayudarte con:\n- Trámites y gestiones administrativas\n- Información sobre reglamentos y normativas\n- Consultas académicas generales\n- Temas de inscripción y regularidad",
            "Me especializo en:\n- Asistencia con trámites administrativos\n- Información sobre reglamentos\n- Consultas académicas\n- Temas de inscripción y regularidad"
        ]
        
        # Si es un agradecimiento, devolver una respuesta alegre
        if is_acknowledgment:
            return random.choice(happy_responses)
            
        # Si es una pregunta sobre capacidades, devolver una respuesta específica
        if is_capabilities:
            return random.choice(capabilities_responses)
        
        # Personalizar el prompt según si tenemos el nombre del usuario
        user_context = f"El usuario se llama {user_name}. " if user_name else ""
        
        prompt = f"""[INST]
Como DrCecim, un asistente virtual de la Facultad de Medicina de la UBA:
- Usa un tono amigable y profesional
- Mantén las respuestas breves y directas
- No hagas preguntas adicionales
- Solo saluda si el usuario está saludando por primera vez
- Si conoces el nombre del usuario, úsalo de manera natural sin forzarlo

{user_context}
Contexto de la consulta: {context}
Consulta del usuario: {query}

Instrucciones específicas:
- Si es un saludo: {"Saluda usando el nombre del usuario si está disponible y menciona que puedes ayudar con trámites y consultas" if is_greeting else "Responde directamente sin saludar"}
- Si es una pregunta de cortesía: {"Responde amablemente mencionando el nombre si está disponible, pero sin volver a presentarte" if is_courtesy else "Responde directamente"}
- Si preguntan sobre tus capacidades: Explica que ayudas con trámites administrativos y consultas académicas
- Si es una consulta médica: Explica amablemente que no puedes responder consultas médicas
- Si preguntan tu identidad: Explica que eres un asistente virtual de la facultad, sin saludar nuevamente

[/INST]"""

        return self.model.generate(prompt)

    def _summarize_previous_message(self, user_id: str) -> str:
        """
        Genera un resumen del mensaje previo para un usuario.
        
        Args:
            user_id (str): ID del usuario
            
        Returns:
            str: Resumen generado
        """
        # Obtener historial del usuario
        history = self.get_user_history(user_id)
        
        if not history or len(history) < 1:
            return "No tengo un mensaje previo para resumir. ¿Puedes hacerme una pregunta específica?"
        
        # Obtener el último mensaje enviado por el bot
        last_query, last_response = history[-1]
        
        # Si el último mensaje es muy corto, no necesita resumen
        if len(last_response) < 150:
            return f"Mi mensaje anterior ya era bastante breve: \"{last_response}\""
        
        # Generar un resumen usando el modelo
        prompt = f"""[INST]
Como DrCecim, asistente virtual de la Facultad de Medicina UBA:

TAREA:
Genera un resumen claro y conciso de tu mensaje anterior.

MENSAJE ORIGINAL:
{last_response}

INSTRUCCIONES:
1. Resume el contenido principal manteniendo la información clave
2. El resumen debe ser aproximadamente 50% más corto que el original
3. Mantén el mismo tono amigable y profesional
4. Incluye los puntos más importantes y relevantes
5. Si hay pasos o instrucciones, presérvales en formato de lista
6. No agregues información nueva que no estaba en el mensaje original
7. No uses frases como "En resumen" o "En conclusión"
[/INST]"""

        try:
            summary = self.model.generate(prompt)
            emoji = random.choice(information_emojis)
            return f"{emoji} {summary}"
        except Exception as e:
            logger.error(f"Error al generar resumen: {str(e)}")
            return "Lo siento, no pude generar un resumen en este momento. ¿Podrías hacerme una pregunta más específica?"

    def process_query(self, query: str, user_id: str = None, user_name: str = None) -> Dict[str, Any]:
        """
        Procesa una consulta utilizando el nuevo sistema de clasificación semántica
        
        Args:
            query (str): La consulta del usuario
            user_id (str, optional): ID del usuario (número de teléfono)
            user_name (str, optional): Nombre del usuario si está disponible
        """
        try:
            # Determinar la intención
            intent, confidence = self._get_query_intent(query)
            logger.info(f"Intención detectada: {intent} (confianza: {confidence:.2f})")
            
            # Si es una referencia a un mensaje anterior
            if intent == 'referencia_anterior':
                if not user_id or not self.get_user_history(user_id):
                    response = "Lo siento, no tengo mensajes previos para resumir. ¿Puedes hacerme una pregunta específica?"
                else:
                    response = self._summarize_previous_message(user_id)
                
                # Actualizar historial
                if user_id:
                    self.update_user_history(user_id, query, response)
                    
                return {
                    "query": query,
                    "response": response,
                    "query_type": intent,
                    "confidence": confidence,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Si es una pregunta sobre el nombre
            if intent == 'pregunta_nombre':
                if user_name:
                    # Lista de respuestas amigables sobre cómo sabemos el nombre
                    name_responses = [
                        f"¡Tu nombre es {user_name}! Lo veo en tu perfil de WhatsApp 😊",
                        f"Me aparece {user_name} en tu perfil de WhatsApp. ¡Un gusto conocerte! 👋",
                        f"¡Te llamas {user_name}! Lo sé porque está en tu perfil de WhatsApp 🤓",
                        f"Puedo ver que te llamas {user_name} por la info de tu perfil. ¡Es un placer! 🌟",
                        f"Tu nombre de perfil es {user_name}. ¡Así es como puedo saludarte personalmente! 😊"
                    ]
                    response = random.choice(name_responses)
                else:
                    response = "Lo siento, en este momento no puedo ver tu nombre en el perfil 😅"
                
                return {
                    "query": query,
                    "response": response,
                    "query_type": intent,
                    "confidence": confidence,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Si es un saludo y tenemos el nombre, personalizar la respuesta
            if intent == 'saludo' and user_name:
                greeting_response = f"{random.choice(greeting_emojis)} ¡Hola {user_name}! Soy DrCecim, el asistente virtual de la Facultad de Medicina. ¿En qué puedo ayudarte?"
                return {
                    "query": query,
                    "response": greeting_response,
                    "query_type": intent,
                    "confidence": confidence,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Si es un agradecimiento, dar una respuesta alegre
            if intent == 'agradecimiento':
                response = self._generate_conversational_response(query, intent, user_name)
                return {
                    "query": query,
                    "response": response,
                    "query_type": intent,
                    "confidence": confidence,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Primero verificar si la consulta corresponde a una FAQ
            response = self._check_faqs(query)
            if response:
                # Si tenemos el nombre y es la primera interacción, personalizamos
                if user_name and not self.get_user_history(user_id):
                    response = f"¡Hola {user_name}! {response}"
                
                logger.info("Consulta respondida desde FAQs")
                return {
                    "query": query,
                    "response": response,
                    "query_type": "faq",
                    "confidence": 1.0,
                    "relevant_chunks": [],
                    "sources": []  # No incluimos fuentes para FAQs
                }
            
            # Si no es una FAQ, continuar con el proceso normal
            if intent in ['saludo', 'pregunta_capacidades', 'identidad', 'cortesia']:
                response = self._generate_conversational_response(query, intent, user_name)
                return {
                    "query": query,
                    "response": response,
                    "query_type": intent,
                    "confidence": confidence,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Manejar consultas médicas
            if intent == 'consulta_medica':
                return {
                    "query": query,
                    "response": self._handle_medical_query(),
                    "query_type": intent,
                    "confidence": confidence,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Para consultas administrativas, académicas y de reglamento
            # continuar con el proceso RAG normal
            # ... resto del código existente para process_query ...

            # Establecer el número de chunks según el tipo de consulta
            num_chunks = {
                'reglamento': 5,  # Más chunks para consultas de reglamento
                'academica': 4,
                'administrativa': 3,
                'consulta_general': 3
            }.get(intent, 3)  # Default a 3 si el tipo no está en el diccionario
            
            logger.info(f"Procesando consulta: {query}")
            
            # Encontrar fragmentos relevantes
            relevant_chunks = self.retrieve_relevant_chunks(query, k=num_chunks)
            
            # Verificar si se encontraron chunks relevantes
            if not relevant_chunks:
                logger.warning("No se encontraron chunks relevantes para la consulta.")
                
                # Verificar si la consulta es sobre sanciones o agresiones
                if intent == 'reglamento':
                    # Intentar una nueva búsqueda con umbral más bajo
                    logger.info("Intentando búsqueda específica con umbral reducido...")
                    self.similarity_threshold = 0.1  # Reducir temporalmente el umbral
                    relevant_chunks = self.retrieve_relevant_chunks(query, k=num_chunks)
                    self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))  # Restaurar umbral
                
                if not relevant_chunks:
                    emoji = random.choice(information_emojis)
                    if intent in ['saludo', 'pregunta_capacidades', 'identidad']:
                        standard_no_info_response = f"{emoji} ¡Hola! Lo siento, no encontré información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas."
                    else:
                        standard_no_info_response = f"{emoji} Lo siento, no encontré información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas."
                
                    return {
                        "query": query,
                        "response": standard_no_info_response,
                        "relevant_chunks": [],
                        "sources": [],
                        "query_type": intent,
                        "confidence": confidence
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
                emoji = random.choice(information_emojis)
                standard_no_info_response = f"{emoji} Lo siento, no encontré información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas."
                
                return {
                    "query": query,
                    "response": standard_no_info_response,
                    "relevant_chunks": [],
                    "sources": [],
                    "query_type": intent,
                    "confidence": confidence
                }
            
            logger.info(f"Se encontraron {len(context_chunks)} fragmentos relevantes de {len(sources)} fuentes")
            
            # Generar respuesta inicial
            response = self.generate_response(query, context, sources)
            
            # Verificar calidad de la respuesta
            verified_response, verification_score = self._verify_response(response, context, intent)
            logger.info(f"Verificación de respuesta completada (score: {verification_score:.2f})")
            
            # Si la verificación indica baja calidad, intentar regenerar
            if verification_score < 0.7:
                logger.warning("Baja calidad de respuesta detectada, intentando regenerar...")
                response = self.generate_response(query, context, sources)
                verified_response, verification_score = self._verify_response(response, context, intent)
            
            # Calcular confianza final
            final_confidence = (confidence + verification_score) / 2
            
            # Actualizar historial del usuario
            if user_id:
                self.update_user_history(user_id, query, response)
            
            return {
                "query": query,
                "response": verified_response,
                "relevant_chunks": relevant_chunks,
                "sources": sources,
                "query_type": intent,
                "confidence": final_confidence
            }
            
        except Exception as e:
            logger.error(f"Error en process_query: {str(e)}", exc_info=True)
            error_response = f"👨‍⚕️ Lo siento, tuve un problema procesando tu consulta. Por favor, intenta de nuevo."
            return {
                "query": query,
                "response": error_response,
                "error": str(e),
                "query_type": "error",
                "confidence": 0.0
            }

    def _check_faqs(self, query: str) -> Optional[str]:
        """
        Verifica si la consulta corresponde a una FAQ y retorna la respuesta correspondiente
        """
        # Palabras clave para cada FAQ
        faq_keywords = {
            "constancia": ["constancia", "alumno regular", "certificado regular"],
            "baja": ["baja", "dar de baja", "darme de baja", "abandonar materia"],
            "anulacion": ["anular", "anulación", "cancelar inscripción", "final"],
            "inscripcion": ["no logro inscribirme", "no salgo asignado", "no me asignan"],
            "reincorporacion": ["reincorporación", "reincorporar", "volver a la carrera"],
            "recursada": ["recursada", "recursar", "segunda vez", "segunda cursada"],
            "tercera": ["tercera cursada", "tercera vez", "3ra cursada"],
            "cuarta": ["cuarta cursada", "cuarta vez", "4ta cursada"],
            "prorroga": ["prórroga", "prorroga", "prorrogar materia"]
        }
        
        # Normalizar la consulta
        query_normalized = unidecode(query.lower())
        
        # Buscar coincidencias
        for faq_type, keywords in faq_keywords.items():
            if any(keyword in query_normalized for keyword in keywords):
                emoji = random.choice(information_emojis)
                if faq_type == "constancia":
                    return f"{emoji} Para tramitar la constancia de alumno regular en el Sitio de Inscripciones:\n- Paso 1: Ingresar tu DNI y contraseña.\n- Paso 2: Seleccionar la opción \"Constancia de alumno regular\" en el inicio de trámites.\n- Paso 3: Imprimir la constancia. Luego, deberás presentarte con tu Libreta Universitaria o DNI y el formulario impreso (1 hoja que posee 3 certificados de alumno regular) en la ventanilla del Ciclo Biomédico."
                elif faq_type == "recursada":
                    return f"{emoji} Para solicitar una recursada, genera el trámite en el Sitio de Inscripciones siguiendo estos pasos:\n- Paso 1: Ingresar tu DNI y contraseña.\n- Paso 2: Seleccionar \"Recursada\".\nEl trámite es automático y, si aparece resuelto positivamente en el sistema, no necesitas acudir a ventanilla.\n- Si en el sistema apareces como dado DE BAJA en la cursada anterior, solo debes generar el trámite y te inscribirás como la primera vez, sin abonar arancel.\n- Si no apareces dado DE BAJA, deberás:\n  1. Realizar el trámite.\n  2. Generar e imprimir el talón de pago.\n  3. Pagar en la Dirección de Tesorería.\n  4. Presentar un comprobante de pago en los buzones del Ciclo Biomédico."
                # ... agregar el resto de las FAQs ...
        
        return None

    def _verify_response(self, response: str, context: str, intent: str) -> tuple:
        """
        Verifica la calidad de la respuesta generada
        """
        # Inicializar el score de verificación
        verification_score = 1.0
        
        # Verificar longitud adecuada
        if len(response) < 50:
            verification_score *= 0.7
        elif len(response) > 500:
            verification_score *= 0.8
        
        # Verificar presencia de información del contexto
        context_keywords = set(context.lower().split())
        response_keywords = set(response.lower().split())
        keyword_overlap = len(context_keywords.intersection(response_keywords))
        
        if keyword_overlap < 5:
            verification_score *= 0.6
        
        # Verificar formato según tipo de consulta
        if intent == 'reglamento':
            if not any(word in response.lower() for word in ['artículo', 'reglamento', 'normativa']):
                verification_score *= 0.8
        elif intent == 'administrativo':
            if not any(word in response.lower() for word in ['trámite', 'pasos', 'procedimiento']):
                verification_score *= 0.8
        
        # Verificar presencia de elementos estructurales
        if not any(emoji in response for emoji in (greeting_emojis + information_emojis)):
            verification_score *= 0.9
        
        if 'fuentes consultadas' not in response.lower():
            verification_score *= 0.9
        
        return response, verification_score

    def get_user_history(self, user_id: str) -> list:
        """Obtiene el historial de un usuario específico"""
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        return self.conversation_histories[user_id]
    
    def update_user_history(self, user_id: str, query: str, response: str):
        """Actualiza el historial de un usuario específico"""
        history = self.get_user_history(user_id)
        history.append((query, response))
        if len(history) > self.max_history_length:
            history = history[-self.max_history_length:]
        self.conversation_histories[user_id] = history

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