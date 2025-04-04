"""
RAG system for the UBA Medicina chatbot.
Retrieves relevant context and generates responses using language models.

This module implements:
1. Context retrieval using embeddings
2. Response generation with language models
3. Vector storage backend integration
4. Memory optimization
5. Error handling and fallbacks

Key features:
- Multiple vector storage backends
- Efficient context retrieval
- Robust error handling
- Memory optimization
- Detailed logging
"""

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

# Cargar variables de entorno
load_dotenv()

# Configuración de logging detallado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar token de Hugging Face si está disponible
if "HUGGING_FACE_HUB_TOKEN" in os.environ:
    huggingface_hub.login(token=os.environ["HUGGING_FACE_HUB_TOKEN"], add_to_git_credential=False)
    logger.info("Configurado token de Hugging Face desde variables de entorno")

# Determinar el entorno (desarrollo o producción)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
logger.info(f"Iniciando sistema RAG en entorno: {ENVIRONMENT}")

def get_device():
    """
    Gets the appropriate device for model inference.
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
        
    Notes:
        - Detects CUDA for NVIDIA GPUs
        - Detects MPS for Apple Silicon
        - Falls back to CPU if neither is available
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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
    """
    Abstract base class for vector storage.
    Defines common interface for different backends.
    """
    
    def search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Searches for similar vectors.
        
        Args:
            query_embedding (List[float]): Query vector
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar vectors with metadata
        """
        raise NotImplementedError

# Implementación para FAISS (desarrollo)
class FAISSVectorStore(VectorStore):
    def __init__(self, index_path: str, metadata_path: str):
        """
        Initializes FAISS vector store.
        
        Args:
            index_path (str): Path to FAISS index file
            metadata_path (str): Path to metadata file
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No se encontró el índice FAISS en {index_path}")
            
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"Índice FAISS cargado con {self.index.ntotal} vectores")
    
    def search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Searches for similar vectors using FAISS.
        
        Args:
            query_embedding (List[float]): Query vector
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar vectors with metadata
        """
        query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    'text': self.metadata[idx]['text'],
                    'score': float(distances[0][i]),
                    'metadata': self.metadata[idx].get('metadata', {})
                })
        return results

# Implementación para Pinecone (producción)
class PineconeVectorStore(VectorStore):
    def __init__(self, index_name: str):
        """
        Initializes Pinecone vector store.
        
        Args:
            index_name (str): Name of Pinecone index
        """
        if index_name not in pinecone.list_indexes():
            raise ValueError(f"No se encontró el índice Pinecone '{index_name}'")
            
        self.index = pinecone.Index(index_name)
        stats = self.index.describe_index_stats()
        logger.info(f"Índice Pinecone '{index_name}' cargado con {stats['total_vector_count']} vectores")
    
    def search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Searches for similar vectors using Pinecone.
        
        Args:
            query_embedding (List[float]): Query vector
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of similar vectors with metadata
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        return [
            {
                'text': match.metadata['text'],
                'score': match.score,
                'metadata': match.metadata
            }
            for match in results.matches
        ]

def get_available_memory_gb():
    """
    Gets available memory in GB.
    
    Returns:
        float: Available memory in GB
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
    Loads model with fallback options for memory constraints.
    
    Args:
        model_path (str): Path to model
        load_kwargs (Dict): Loading parameters
        
    Returns:
        tuple: (model, tokenizer)
        
    Notes:
        - Tries different quantization options
        - Falls back to CPU if needed
        - Handles memory errors gracefully
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logger.warning(f"Error loading model with parameters: {str(e)}")
        
        # Try with 8-bit quantization
        try:
            load_kwargs['load_in_8bit'] = True
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            logger.warning(f"Error loading model with 8-bit quantization: {str(e)}")
            
            # Try with 4-bit quantization
            try:
                load_kwargs['load_in_4bit'] = True
                load_kwargs.pop('load_in_8bit', None)
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                return model, tokenizer
            except Exception as e:
                logger.error(f"Error loading model with 4-bit quantization: {str(e)}")
                raise

# Clase base para modelos
class BaseModel:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

# Clase para usar la API de Hugging Face
class APIModel(BaseModel):
    def __init__(self, model_name: str, api_token: str, timeout: int = 30):
        """
        Initializes API model.
        
        Args:
            model_name (str): Model name/endpoint
            api_token (str): API authentication token
            timeout (int): Request timeout in seconds
        """
        self.model_name = model_name
        self.api_token = api_token
        self.timeout = timeout
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text using API.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            **kwargs
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error generating text with API: {str(e)}")
            raise

# Clase para modelo local
class LocalModel(BaseModel):
    def __init__(self, model, tokenizer, device):
        """
        Initializes local model.
        
        Args:
            model: Hugging Face model
            tokenizer: Hugging Face tokenizer
            device (str): Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text using local model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            outputs = self.model.generate(
                **inputs,
                **kwargs
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating text with local model: {str(e)}")
            raise

class RAGSystem:
    def __init__(
        self,
        model_path: str = os.getenv('BASE_MODEL_PATH', 'models/finetuned_model'),
        embeddings_dir: str = os.getenv('EMBEDDINGS_DIR', 'data/embeddings'),
        device: str = os.getenv('DEVICE', 'mps')
    ):
        """
        Initializes RAG system.
        
        Args:
            model_path (str): Path to language model
            embeddings_dir (str): Directory with embeddings
            device (str): Device for inference
            
        Notes:
        - Loads model and embeddings
        - Configures vector store
        - Sets up logging
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
        embedding_model_name = os.getenv('EMBEDDING_MODEL', 
                                       'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Inicializar el almacén vectorial
        self.vector_store = self._initialize_vector_store()

    def _load_local_model(self, model_path: str):
        """
        Loads local language model.
        
        Args:
            model_path (str): Path to model
        """
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        self.model, self.tokenizer = load_model_with_fallback(model_path, load_kwargs)
        
    def _initialize_api_model(self) -> APIModel:
        """
        Initializes API-based model.
        
        Returns:
            APIModel: Initialized API model
        """
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
        Initializes vector store based on environment.
        
        Returns:
            VectorStore: Initialized vector store
            
        Notes:
        - Uses FAISS for development
        - Uses Pinecone for production
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
        metadata_path = str(self.embeddings_dir / 'metadata.json')
        return FAISSVectorStore(index_path, metadata_path)
        
    def retrieve_relevant_chunks(
        self,
        query: str,
        k: int = None
    ) -> List[Dict]:
        """
        Retrieves relevant text chunks for query.
        
        Args:
            query (str): User query
            k (int, optional): Number of chunks to retrieve
            
        Returns:
            List[Dict]: Relevant chunks with metadata
        """
        if k is None:
            k = int(os.getenv('NUM_CHUNKS', '3'))
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search for similar chunks
        results = self.vector_store.search(query_embedding, k)
        
        return results
        
    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        """
        Generates response using retrieved context.
        
        Args:
            query (str): User query
            context (str): Retrieved context
            sources (List[str], optional): Source documents
            
        Returns:
            str: Generated response
        """
        try:
            # Lista de emojis para enriquecer las respuestas
            information_emojis = ["📚", "📖", "ℹ️", "📊", "🔍", "📝", "📋", "📈", "📌", "🧠"]
            greeting_emojis = ["👋", "😊", "🤓", "👨‍⚕️", "👩‍⚕️", "🎓", "🌟"]
            
            # Seleccionar emojis de manera pseudo-aleatoria pero consistente
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            info_emoji = information_emojis[query_hash % len(information_emojis)]
            greeting_emoji = greeting_emojis[query_hash % len(greeting_emojis)]
            
            # Verificar si es una consulta sobre el nombre del bot
            name_queries = [
                "cómo te llamás", "como te llamas", "¿cómo te llamas?", "¿como te llamas?", "cómo te llamas?", "como te llamas?",
                "cuál es tu nombre", "cual es tu nombre", "¿cuál es tu nombre?", "¿cual es tu nombre?", "cuál es tu nombre?", "cual es tu nombre?",
                "quién eres", "quien eres", "¿quién eres?", "¿quien eres?", "quién eres?", "quien eres?",
                "cómo te dicen", "como te dicen", "¿cómo te dicen?", "¿como te dicen?", "cómo te dicen?", "como te dicen?",
                "tu nombre", "cómo te llaman", "como te llaman",
                "cuál es tu apellido", "cual es tu apellido"
            ]
            
            # Verificación más estricta para consultas sobre el nombre
            is_name_query = False
            clean_query = query.lower().strip()
            if clean_query in name_queries:
                is_name_query = True
            else:
                # Si no hay coincidencia exacta, intentar coincidencia parcial
                is_name_query = any(phrase in clean_query for phrase in name_queries)
            
            # Verificación adicional para "Cómo te llamas?" exactamente como en WhatsApp
            if "cómo te llamas" in clean_query or "como te llamas" in clean_query:
                is_name_query = True
                
            print(f"DEBUG - Query: '{query}', is_name_query: {is_name_query}")  # Depuración
            
            # Lista de palabras de saludo
            greeting_words = ['hola', 'buenas', 'buen día', 'buen dia', 'buenos días', 'buenos dias', 
                              'buenas tardes', 'buenas noches', 'saludos', 'que tal', 'qué tal', 'como va', 'cómo va']
            
            # Identificar si hay un saludo en la consulta
            has_greeting = False
            greeting_used = None
            for word in greeting_words:
                if word in query.lower().strip().split() or query.lower().strip().startswith(word):
                    has_greeting = True
                    greeting_used = word
                    break
            
            # Caso especial para "buenas" que puede estar al inicio sin espacio
            if not has_greeting and (query.lower().strip().startswith('buenas')):
                has_greeting = True
                greeting_used = 'buenas'
            
            # Determinar si es solo un saludo sin pregunta
            is_greeting_only = query.lower().strip() in greeting_words or any(
                query.lower().strip() == word or query.lower().strip().startswith(word + " ")
                for word in greeting_words
            )
            
            # Determinar el saludo a usar en la respuesta si hay uno en la consulta
            greeting_prefix = ""
            if has_greeting:
                if greeting_used in ['hola', 'saludos']:
                    greeting_prefix = f"👨‍⚕️ ¡Hola! Soy DrCecim. "
                elif greeting_used in ['buenas', 'buen día', 'buen dia', 'buenos días', 'buenos dias']:
                    greeting_prefix = f"👨‍⚕️ ¡Buenos días! Soy DrCecim. "
                elif greeting_used == 'buenas tardes':
                    greeting_prefix = f"👨‍⚕️ ¡Buenas tardes! Soy DrCecim. "
                elif greeting_used == 'buenas noches':
                    greeting_prefix = f"👨‍⚕️ ¡Buenas noches! Soy DrCecim. "
                elif greeting_used in ['qué tal', 'que tal', 'cómo va', 'como va']:
                    greeting_prefix = f"👨‍⚕️ ¿Cómo va? Soy DrCecim. "
                else:
                    greeting_prefix = f"👨‍⚕️ ¡Buenas! Soy DrCecim. "
            else:
                greeting_prefix = f"👨‍⚕️ "
            
            # Si es solo un saludo, responder directamente sin buscar embeddings
            if is_greeting_only:
                greeting_responses = [
                    f"👨‍⚕️ ¡Hola! Soy DrCecim, tu asistente de la Facultad de Medicina. ¿En qué puedo ayudarte hoy?",
                    f"👨‍⚕️ ¡Buenas! Soy DrCecim, ¿en qué puedo asistirte hoy?",
                    f"👨‍⚕️ ¿Cómo va? Soy DrCecim, tu asistente académico. ¿Con qué puedo ayudarte?",
                    f"👨‍⚕️ Hola, soy DrCecim. ¿En qué puedo orientarte hoy?",
                    f"👨‍⚕️ Saludos. Soy DrCecim, ¿necesitas ayuda con algún tema en particular?",
                    f"👨‍⚕️ ¡Buen día! Soy DrCecim, asistente de la Facultad de Medicina. ¿En qué puedo colaborar?"
                ]
                import random
                return {
                    "query": query,
                    "response": random.choice(greeting_responses),
                    "relevant_chunks": [],
                    "sources": []
                }
            
            if not context or context.strip() == "":
                # Respuesta para cuando no hay contexto relevante
                system_prompt = f"""Eres un asistente virtual especializado de la Facultad de Medicina de la Universidad de Buenos Aires. 
                Tu tono es amable, profesional e incluyes emojis apropiados en tus respuestas.
                Hablas directamente con los alumnos de medicina, no con profesores.
                
                ### Estilo de respuesta:
                - Usa un tono formal pero amigable
                - Incluye al menos un emoji relevante en cada respuesta
                - Preséntate como "DrCecim" SOLO si la consulta incluye un saludo
                - Sé conciso y directo
                
                ### Ejemplos de respuestas:
                Pregunta con saludo: Hola, ¿cuándo comienzan las inscripciones?
                Respuesta: {greeting_emoji} Soy DrCecim. Las inscripciones comienzan el 15 de marzo. ¡No olvides tener toda tu documentación lista! {info_emoji}
                
                Pregunta sin saludo: ¿Qué carreras ofrece la facultad?
                Respuesta: {greeting_emoji} La facultad ofrece las siguientes carreras: Medicina, Enfermería, Kinesiología, Nutrición y Obstetricia. {info_emoji} ¿Necesitas información específica sobre alguna?
                """
                
                user_prompt = f"No tengo información específica sobre: '{query}'. Responde amablemente que no tienes información suficiente sobre este tema y sugiere preguntar sobre otros temas relacionados con la universidad."
                
                # Formato del prompt según si es API o modelo local
                prompt = f"{system_prompt}\n\n{user_prompt}"
                
                response = self.model.generate(prompt, max_length=512, temperature=0.7)
                
                # Asegurar que la respuesta incluya emojis y la presentación correcta
                if has_greeting and "DrCecim" not in response:
                    response = f"{greeting_emoji} Soy DrCecim. {response}"
                elif not has_greeting and "DrCecim" in response:
                    # Si no hay saludo, eliminar la mención a DrCecim
                    response = re.sub(r'(?i)(Soy DrCecim\.?|DrCecim aquí\.?|DrCecim:)\s*', f'👨‍⚕️ ', response)
                
                if not any(emoji in response for emoji in information_emojis + greeting_emojis):
                    response = f"{response} {info_emoji}"
                
                return response
            
            # Si hay contexto relevante, usar el contexto y las fuentes
            if sources:
                fuentes_str = ", ".join(sources)
                sources_context = f"Fuentes consultadas: {fuentes_str}"
            else:
                sources_context = ""
            
            # Instrucción específica para el saludo
            greeting_instruction = ""
            if has_greeting:
                greeting_instruction = f"Inicia tu respuesta con '{greeting_prefix}'"
            else:
                greeting_instruction = f"Inicia tu respuesta con '{greeting_prefix}' sin mencionar el nombre DrCecim"
            
            # Aplicar técnicas de prompt engineering de Mistral AI
            system_prompt = f"""Eres un asistente virtual especializado, llamado DrCecim, de la Facultad de Medicina de la Universidad de Buenos Aires.

### ADVERTENCIA EXTREMADAMENTE IMPORTANTE:
NUNCA GENERES PREGUNTAS Y RESPUESTAS. RESPONDE SOLO Y EXCLUSIVAMENTE A LA CONSULTA DEL USUARIO. SOLO GENERA UNA ÚNICA RESPUESTA. NO CREES DIÁLOGOS, NI CONVERSACIONES, NI INTERACCIONES ADICIONALES.

### INSTRUCCIÓN SOBRE SALUDOS - MUY IMPORTANTE:
La consulta del usuario {'' if has_greeting else 'NO'} contiene un saludo.
{'DEBES iniciar tu respuesta incluyendo "Soy DrCecim" en tu saludo.' if has_greeting else 'NO debes mencionar tu nombre "DrCecim" en tu respuesta.'}

### Instrucciones de Formato:
1. Usa un tono amable y profesional.
2. SIEMPRE empieza tus mensajes con el emoji 👨‍⚕️.
3. Preséntate como "DrCecim" SOLO si el usuario te saluda primero.
4. Cuando necesites crear listas:
   - Usa el formato exacto de WhatsApp: guion + espacio + texto + espacio + emoji
   - Un elemento por línea, sin texto después del emoji
   - Ejemplo correcto: "- Elemento uno 📝"
   - NUNCA incluyas texto después del emoji
   - NUNCA escribas algo como: "- Elemento 📝 información adicional"

### Formato específico para WhatsApp:
- Usa guiones (-) para listas, NUNCA asteriscos o bullets (•)
- Coloca el emoji AL FINAL de cada línea de lista, NUNCA al principio
- Después de cada emoji en la lista, usa un SALTO DE LÍNEA completo
- Mantén los elementos de la lista CORTOS y SIMPLES

### Ejemplos CORRECTOS de listas para WhatsApp:
👨‍⚕️ Las sanciones que se pueden aplicar son:
- Apercibimiento o suspensión de hasta un año ⚠️
- Suspensión de uno a cinco años ⚠️
- Expulsión definitiva ⚠️

### {'Ejemplos de respuestas CON saludo:' if has_greeting else 'Ejemplos de respuestas SIN saludo:'}
{'👨‍⚕️ ¡Buenas! Soy DrCecim. Las sanciones que se pueden aplicar son...' if has_greeting else '👨‍⚕️ Las sanciones que se pueden aplicar son...'}
{'👨‍⚕️ ¿Cómo va? Soy DrCecim. La información que solicitaste es...' if has_greeting else '👨‍⚕️ La información que solicitaste es...'}

### REGLAS CRÍTICAS - LEE ESTO CUIDADOSAMENTE:
1. SOLO RESPONDE UNA VEZ A LA PREGUNTA ESPECÍFICA DEL USUARIO. NO GENERES NINGUNA PREGUNTA NI RESPUESTA ADICIONAL.
2. NO CREES DIÁLOGOS FICTICIOS BAJO NINGUNA CIRCUNSTANCIA.
3. NO INVENTES PREGUNTAS. SI VES QUE ESTÁS CREANDO UNA PREGUNTA, DETENTE INMEDIATAMENTE.
4. NO RESPONDAS A PREGUNTAS QUE EL USUARIO NO TE HA HECHO EXPLÍCITAMENTE.
5. SOLO PROPORCIONA INFORMACIÓN DIRECTAMENTE RELACIONADA CON LA CONSULTA DEL USUARIO.
6. NO INCLUYAS NINGUNA PREGUNTA EN TU RESPUESTA.
7. CADA MENSAJE TUYO DEBE CONTENER UNA ÚNICA RESPUESTA CONCISA.
8. NUNCA CREES TEXTO QUE EMPIECE CON SIGNOS DE INTERROGACIÓN (¿).

<contexto>
{context}
</contexto>

<consulta>
{query}
</consulta>

Responde ÚNICAMENTE a la consulta con información del contexto. SOLO GENERA UNA RESPUESTA. NO HAGAS PREGUNTAS."""
            
            user_prompt = f"Responde de manera directa y concisa a la consulta. Si necesitas hacer una lista, usa EXACTAMENTE el formato indicado en las instrucciones."
            
            # Formato completo del prompt
            prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.model.generate(prompt, max_length=512, temperature=0.7)
            
            # Post-procesar respuesta para asegurar el formato correcto
            response = re.sub(r'(Según el contexto|Basado en el contexto|De acuerdo con el contexto|Como se menciona en el contexto|Respuesta:|Según la información proporcionada)', '', response, flags=re.IGNORECASE)
            response = re.sub(r'<[^>]*>', '', response)  # Eliminar etiquetas HTML
            response = response.strip()
            
            # Eliminar preguntas autogeneradas y sus respuestas
            # Esto dividirá la respuesta en la primera oración que termine con punto 
            # o en el primer párrafo, lo que sea más corto
            # Buscar la primera pregunta (si existe) y cortar todo lo que sigue
            question_pattern = r'(?:\n|^)(?:\?|¿).*\?'
            match = re.search(question_pattern, response)
            if match:
                response = response[:match.start()]
                
            # Si después de eliminar preguntas, queda una respuesta con párrafos múltiples
            # mantener solo el primer párrafo relevante
            paragraphs = re.split(r'\n\s*\n', response)
            if len(paragraphs) > 1:
                # Mantener el primer párrafo (respuesta principal) y cualquier lista que pueda seguir
                main_content = paragraphs[0]
                if '- ' in main_content or main_content.strip().endswith(':'):
                    # Si el primer párrafo contiene una lista o termina con dos puntos,
                    # incluir también el siguiente párrafo que probablemente contiene la lista
                    for p in paragraphs[1:]:
                        if p.strip().startswith('- '):
                            main_content += '\n\n' + p
                response = main_content
            
            # Asegurar que las listas usen guiones y no bullets
            response = response.replace('• ', '- ')
            
            # Detectar si hay listas en la respuesta
            has_bullet_list = '- ' in response
            
            # Procesar formatos de lista con viñetas para mejorar presentación
            if has_bullet_list:
                # Asegurarse de que cada elemento de lista esté en una línea separada
                response = re.sub(r'([^\n])(\s*-\s+)', r'\1\n- ', response)
                
                # Separar cualquier texto que venga después de un emoji en una línea de lista
                lines = response.split('\n')
                processed_lines = []
                
                for line in lines:
                    if line.strip().startswith('- '):
                        # Buscar emojis en la línea
                        emoji_pattern = r'(📚|📖|ℹ️|📊|🔍|📝|📋|📈|📌|🧠|👋|😊|🤓|👨‍⚕️|👩‍⚕️|🎓|🌟|📄|📅|🗓️|⚠️)'
                        emoji_match = re.search(emoji_pattern, line)
                        
                        if emoji_match:
                            emoji_pos = emoji_match.end()
                            # Si hay texto después del emoji, separarlo
                            if emoji_pos < len(line) and line[emoji_pos:].strip():
                                processed_lines.append(line[:emoji_pos])  # Línea con el ítem y el emoji
                                processed_lines.append("")  # Línea en blanco para separar
                                processed_lines.append(line[emoji_pos:].strip())  # Texto adicional como párrafo
                            else:
                                processed_lines.append(line)
                        else:
                            processed_lines.append(line)
                    else:
                        processed_lines.append(line)
                
                response = '\n'.join(processed_lines)
            
            # Asegurar que comienza con el emoji de doctor
            if not response.startswith("👨‍⚕️"):
                response = "👨‍⚕️ " + response.lstrip()
            
            # Asegurar que la respuesta tenga el formato adecuado con el saludo correcto
            if has_greeting and "DrCecim" not in response and not response.startswith(greeting_prefix):
                if response.startswith("👨‍⚕️"):
                    # Si ya empieza con el emoji, reemplazar con el greeting_prefix completo
                    response = greeting_prefix + response[5:].lstrip()
                else:
                    response = greeting_prefix + response
            elif not has_greeting and "DrCecim" in response:
                # Si no hay saludo, eliminar menciones a DrCecim
                response = re.sub(r'(?i)(Soy DrCecim\.?|DrCecim aquí\.?|DrCecim:)\s*', f'👨‍⚕️ ', response)
                
                # También eliminar saludos si no hubo saludo en la consulta
                response = re.sub(r'(?i)(¡Hola!|Hola,|¡Buenos días!|Buenos días|¡Buenas tardes!|Buenas tardes|¡Buenas noches!|Buenas noches|Saludos|¡Buenas!|Buenas,)\s*', '', response)
            
            # Eliminar líneas vacías duplicadas
            response = re.sub(r'\n\s*\n+', '\n\n', response)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error al generar respuesta: {str(e)}")
            if has_greeting:
                return f"👨‍⚕️ Soy DrCecim. Lo siento, tuve un problema procesando tu consulta. Por favor, intenta de nuevo."
            else:
                return f"👨‍⚕️ Lo siento, tuve un problema procesando tu consulta. Por favor, intenta de nuevo."
        
    def process_query(self, query: str, num_chunks: int = None) -> Dict:
        """
        Processes user query end-to-end.
        
        Args:
            query (str): User query
            num_chunks (int, optional): Number of chunks to retrieve
            
        Returns:
            Dict: Response with metadata
        """
        try:
            # Lista de emojis para enriquecer las respuestas
            information_emojis = ["📚", "📖", "ℹ️", "📊", "🔍", "📝", "📋", "📈", "📌", "🧠"]
            greeting_emojis = ["👋", "😊", "🤓", "👨‍⚕️", "👩‍⚕️", "🎓", "🌟"]
            
            # Seleccionar emojis de manera pseudo-aleatoria pero consistente
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            info_emoji = information_emojis[query_hash % len(information_emojis)]
            greeting_emoji = greeting_emojis[query_hash % len(greeting_emojis)]
            
            # Lista de palabras de saludo
            greeting_words = ['hola', 'buenas', 'buen día', 'buen dia', 'buenos días', 'buenos dias', 
                              'buenas tardes', 'buenas noches', 'saludos', 'que tal', 'qué tal', 'como va', 'cómo va']
            
            # Verificar si es una consulta sobre el nombre del bot
            name_queries = [
                "cómo te llamás", "como te llamas", "¿cómo te llamas?", "¿como te llamas?", "cómo te llamas?", "como te llamas?",
                "cuál es tu nombre", "cual es tu nombre", "¿cuál es tu nombre?", "¿cual es tu nombre?", "cuál es tu nombre?", "cual es tu nombre?",
                "quién eres", "quien eres", "¿quién eres?", "¿quien eres?", "quién eres?", "quien eres?",
                "cómo te dicen", "como te dicen", "¿cómo te dicen?", "¿como te dicen?", "cómo te dicen?", "como te dicen?",
                "tu nombre", "cómo te llaman", "como te llaman",
                "cuál es tu apellido", "cual es tu apellido"
            ]
            
            # Verificación más estricta para consultas sobre el nombre
            is_name_query = False
            clean_query = query.lower().strip()
            if clean_query in name_queries:
                is_name_query = True
            else:
                # Si no hay coincidencia exacta, intentar coincidencia parcial
                is_name_query = any(phrase in clean_query for phrase in name_queries)
            
            # Verificación adicional para "Cómo te llamas?" exactamente como en WhatsApp
            if "cómo te llamas" in clean_query or "como te llamas" in clean_query:
                is_name_query = True
                
            print(f"DEBUG - Query: '{query}', is_name_query: {is_name_query}")  # Depuración
            
            # Si pregunta por el nombre, responder directamente
            if is_name_query:
                name_response = f"👨‍⚕️ Me llamo DrCecim. Soy un asistente virtual especializado en información académica de la Facultad de Medicina de la Universidad de Buenos Aires."
                
                return {
                    "query": query,
                    "response": name_response,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # 2. SEGUNDO: Verificar si es una consulta sobre las capacidades del bot
            meta_queries = [
                "qué hace", "que hace", "qué podés hacer", "que podes hacer",
                "en qué me podés ayudar", "en que me podes ayudar",
                "cómo me podés ayudar", "como me podes ayudar",
                "qué información tenés", "que informacion tenes",
                "qué información conocés", "que informacion conoces",
                "qué sabés", "que sabes", "para qué servís", "para que servis",
                "qué tipo de consulta", "que tipo de consulta",
                "qué tipo de información", "que tipo de informacion",
                "qué tipo de preguntas", "que tipo de preguntas",
                "qué consultas puedo hacer", "que consultas puedo hacer",
                "qué me podés decir", "que me podes decir",
                "qué puedo consultarte", "que puedo consultarte",
                "qué servicios ofrece", "que servicios ofrece",
                "cuáles son tus funciones", "cuales son tus funciones",
                "sobre qué temas", "sobre que temas", "qué temas", "que temas",
                "temas de consulta", "qué materias", "que materias",
                "de qué me podés informar", "de que me podes informar", "qué me puedes informar",
                "sobre qué podés ayudarme", "sobre que podes ayudarme",
                "qué me puedes informar", "que puedes decirme", "qué puedes decirme", 
                "que puedo preguntarte", "qué puedo preguntarte"
            ]
            
            # Mejor detección de meta-queries
            is_meta_query = False
            clean_query = query.lower().strip()
            
            # Comprobar coincidencia exacta primero
            if clean_query in meta_queries:
                is_meta_query = True
            else:
                # Si no hay coincidencia exacta, buscar coincidencias parciales
                for phrase in meta_queries:
                    if phrase in clean_query:
                        is_meta_query = True
                        break
                        
                # Verificación adicional para consultas como "¿Qué me puedes informar?"
                if re.search(r'(?:qué|que).*(?:pued(?:o|es)|pod(?:és|es)).*(?:informar|ayudar|consultar|preguntar)', clean_query):
                    is_meta_query = True
            
            if is_meta_query:
                meta_response = f"👨‍⚕️ Puedo ayudarte con consultas sobre:\n- Reglamento académico de la Facultad de Medicina 📚\n- Condiciones de regularidad para alumnos 📋\n- Régimen disciplinario y sanciones 🎓\n- Trámites administrativos para estudiantes 📄\n- Requisitos académicos y normativas 📌"
                
                return {
                    "query": query,
                    "response": meta_response,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # 3. TERCERO: Identificar si hay un saludo en la consulta
            has_greeting = False
            greeting_used = None
            for word in greeting_words:
                if word in query.lower().split() or query.lower().strip().startswith(word):
                    has_greeting = True
                    greeting_used = word
                    break
            
            # Caso especial para "buenas" que puede estar al inicio sin espacio
            if not has_greeting and (query.lower().strip().startswith('buenas')):
                has_greeting = True
                greeting_used = 'buenas'
            
            # Determinar si es solo un saludo sin pregunta
            is_greeting_only = query.lower().strip() in greeting_words or any(
                query.lower().strip() == word or query.lower().strip().startswith(word + " ")
                for word in greeting_words
            )
            
            # Determinar el saludo a usar en la respuesta si hay uno en la consulta
            greeting_prefix = ""
            if has_greeting:
                if greeting_used in ['hola', 'saludos']:
                    greeting_prefix = f"👨‍⚕️ ¡Hola! Soy DrCecim. "
                elif greeting_used in ['buenas', 'buen día', 'buen dia', 'buenos días', 'buenos dias']:
                    greeting_prefix = f"👨‍⚕️ ¡Buenos días! Soy DrCecim. "
                elif greeting_used == 'buenas tardes':
                    greeting_prefix = f"👨‍⚕️ ¡Buenas tardes! Soy DrCecim. "
                elif greeting_used == 'buenas noches':
                    greeting_prefix = f"👨‍⚕️ ¡Buenas noches! Soy DrCecim. "
                elif greeting_used in ['qué tal', 'que tal', 'cómo va', 'como va']:
                    greeting_prefix = f"👨‍⚕️ ¿Cómo va? Soy DrCecim. "
                else:
                    greeting_prefix = f"👨‍⚕️ ¡Buenas! Soy DrCecim. "
            else:
                greeting_prefix = f"👨‍⚕️ "
            
            # Si es solo un saludo, responder directamente sin buscar embeddings
            if is_greeting_only:
                greeting_responses = [
                    f"👨‍⚕️ ¡Hola! Soy DrCecim, tu asistente de la Facultad de Medicina. ¿En qué puedo ayudarte hoy?",
                    f"👨‍⚕️ ¡Buenas! Soy DrCecim, ¿en qué puedo asistirte hoy?",
                    f"👨‍⚕️ ¿Cómo va? Soy DrCecim, tu asistente académico. ¿Con qué puedo ayudarte?",
                    f"👨‍⚕️ Hola, soy DrCecim. ¿En qué puedo orientarte hoy?",
                    f"👨‍⚕️ Saludos. Soy DrCecim, ¿necesitas ayuda con algún tema en particular?",
                    f"👨‍⚕️ ¡Buen día! Soy DrCecim, asistente de la Facultad de Medicina. ¿En qué puedo colaborar?"
                ]
                import random
                return {
                    "query": query,
                    "response": random.choice(greeting_responses),
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # 5. QUINTO: Para preguntas normales, seguir el flujo habitual de RAG
            if num_chunks is None:
                num_chunks = int(os.getenv('NUM_CHUNKS', 3))
            
            # Logging de la consulta para debugging
            logger.info(f"Procesando consulta: '{query}'")
            
            # Encontrar fragmentos relevantes
            relevant_chunks = self.retrieve_relevant_chunks(query, k=num_chunks)
            
            # Verificar si se encontraron chunks relevantes
            if not relevant_chunks:
                logger.warning("No se encontraron chunks relevantes para la consulta.")
                # Respuesta estándar cuando no hay información disponible
                if has_greeting:
                    standard_no_info_response = f"👨‍⚕️ ¡Hola! Soy DrCecim. No tengo información suficiente sobre esto en mis documentos. Si necesitas información específica sobre otro tema relacionado con la Facultad de Medicina, no dudes en preguntar. ¡Estoy aquí para ayudarte!"
                else:
                    standard_no_info_response = f"👨‍⚕️ No tengo información suficiente sobre esto en mis documentos. Si necesitas información específica sobre otro tema relacionado con la Facultad de Medicina, no dudes en preguntar. ¡Estoy aquí para ayudarte!"
                
                return {
                    "query": query,
                    "response": standard_no_info_response,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Construir contexto de manera segura, incluyendo la fuente de cada fragmento
            context_chunks = []
            sources = []
            
            # Establecer un umbral de relevancia mínima - mucho más permisivo ahora
            relevance_threshold = float(os.getenv('RELEVANCE_THRESHOLD', '20.0'))  # Aumentado desde 17.0 a 20.0
            has_relevant_content = False
            
            for i, chunk in enumerate(relevant_chunks):
                # Verificar si el chunk tiene contenido
                chunk_has_content = False
                content = ""
                
                if "text" in chunk and chunk["text"].strip():
                    content = chunk["text"]
                    chunk_has_content = True
                else:
                    logger.warning(f"Chunk sin contenido válido: {chunk}")
                    continue  # Saltar este chunk
                
                # Evaluar si el chunk es realmente relevante para la consulta
                if 'score' in chunk:
                    score = chunk['score']
                    logger.info(f"Chunk {i+1} score: {score}")
                    if score < relevance_threshold:
                        has_relevant_content = True
                        logger.info(f"Chunk {i+1} es relevante (score: {score})")
                
                # Obtener información de la fuente
                source = ""
                if "metadata" in chunk and chunk["metadata"]:
                    source = chunk["metadata"].get('source', '')
                    if source and source not in sources:
                        sources.append(source)
                
                # Solo agregar chunks con contenido válido
                if chunk_has_content:
                    # Formatear el fragmento con su fuente pero sin usar FRAGMENTO en el mensaje
                    # para evitar que el modelo lo copie en la respuesta
                    formatted_chunk = f"Información de {source}:\n{content}"
                    context_chunks.append(formatted_chunk)
                    logger.info(f"Agregado chunk relevante de {source} (score: {chunk.get('score', 'N/A')})")
            
            # Unir los chunks para formar el contexto
            context = '\n\n'.join(context_chunks)
            
            # Si no hay contexto después de filtrar o no hay contenido relevante, usar mensaje informativo
            if not context.strip() or not has_relevant_content:
                logger.warning("No se encontró contexto suficientemente relevante para la consulta.")
                # Respuesta estándar cuando no hay información disponible
                if has_greeting:
                    standard_no_info_response = f"👨‍⚕️ ¡Hola! Soy DrCecim. No tengo información suficiente sobre esto en mis documentos. Si necesitas información específica sobre otro tema relacionado con la Facultad de Medicina, no dudes en preguntar. ¡Estoy aquí para ayudarte!"
                else:
                    standard_no_info_response = f"👨‍⚕️ No tengo información suficiente sobre esto en mis documentos. Si necesitas información específica sobre otro tema relacionado con la Facultad de Medicina, no dudes en preguntar. ¡Estoy aquí para ayudarte!"
                
                return {
                    "query": query,
                    "response": standard_no_info_response,
                    "relevant_chunks": [],
                    "sources": []
                }
            
            logger.info(f"Se encontraron {len(context_chunks)} fragmentos relevantes de {len(sources)} fuentes: {', '.join(sources)}")
            
            # Generar respuesta con el contexto y las fuentes
            response = self.generate_response(query, context, sources)
            
            # Asegurar que la respuesta tenga el formato adecuado
            if has_greeting and "DrCecim" not in response:
                # Verificar si la respuesta ya tiene el emoji del doctor
                if response.startswith("👨‍⚕️"):
                    # Reemplazar el emoji con el saludo completo
                    response = greeting_prefix + response[5:].lstrip()
                else:
                    response = greeting_prefix + response
            elif not has_greeting and "DrCecim" in response:
                # Si no hay saludo, eliminar menciones a DrCecim
                response = re.sub(r'(?i)(Soy DrCecim\.?|DrCecim aquí\.?|DrCecim:)\s*', f'👨‍⚕️ ', response)
            
            if not any(emoji in response for emoji in information_emojis + greeting_emojis):
                response = f"{response} {info_emoji}"
            
            # Agregar fuente de información si hay sources
            final_response = response
            if sources and len(sources) > 0:
                # Limpiar nombres de fuentes (quitar .pdf, guiones bajos, etc.)
                clean_sources = []
                for source in sources:
                    # Reemplazar guiones bajos y guiones con espacios
                    clean_source = source.replace('_', ' ').replace('-', ' ')
                    clean_sources.append(clean_source)
                
                # Agregar fuente al final del mensaje
                if len(clean_sources) == 1:
                    final_response = f"{response}\n\nEsta información la puedes encontrar en: {clean_sources[0]}"
                else:
                    sources_text = ", ".join(clean_sources)
                    final_response = f"{response}\n\nEsta información la puedes encontrar en: {sources_text}"
            
            return {
                "query": query,
                "response": final_response,
                "relevant_chunks": relevant_chunks,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error en process_query: {str(e)}", exc_info=True)
            if has_greeting:
                error_response = f"👨‍⚕️ Soy DrCecim. Lo siento, tuve un problema procesando tu consulta. Por favor, intenta de nuevo."
            else:
                error_response = f"👨‍⚕️ Lo siento, tuve un problema procesando tu consulta. Por favor, intenta de nuevo."
            
            return {
                "query": query,
                "response": error_response,
                "error": str(e)
            }

def main():
    """
    Main function for testing RAG system.
    """
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
                if 'source' in chunk:
                    print(f"- {chunk['source']} (score: {chunk['score']:.3f})")
                else:
                    print(f"- {chunk.get('id', 'unknown')}")
        except Exception as e:
            logger.error(f"Error al procesar la consulta: {str(e)}")
            print("Lo siento, hubo un error al procesar tu consulta.")

if __name__ == "__main__":
    main() 