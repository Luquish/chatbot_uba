import os
import logging
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union
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

# Configuración de logging
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
    
    def search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """
        Búsqueda de vectores similares en FAISS.
        
        Args:
            query_embedding (List[float]): Embedding de la consulta
            k (int): Número de resultados a retornar
            
        Returns:
            List[Dict]: Lista de resultados con metadatos
        """
        # Convertir a numpy y formato correcto
        query_embedding_np = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        # Realizar búsqueda
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Obtener metadatos y resultados
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                metadata = self.metadata.iloc[idx].to_dict()
                metadata['distance'] = float(distance)
                results.append(metadata)
                
        return results

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
        embedding_model_name = os.getenv('EMBEDDING_MODEL', 
                                       'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Inicializar el almacén vectorial
        self.vector_store = self._initialize_vector_store()

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
        
    def retrieve_relevant_chunks(
        self,
        query: str,
        k: int = None
    ) -> List[Dict]:
        """
        Recupera los chunks más relevantes para una consulta.
        
        Args:
            query (str): Consulta del usuario
            k (int): Número de chunks a recuperar
            
        Returns:
            List[Dict]: Lista de chunks relevantes con sus metadatos
        """
        if k is None:
            k = int(os.getenv('RAG_NUM_CHUNKS', '3'))
            
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Buscar chunks similares en el almacén vectorial
        search_results = self.vector_store.search(query_embedding, k * 10)  # Buscamos muchos más chunks para tener mayor probabilidad de encontrar relevantes
        
        # Imprimir los primeros resultados para debugging
        logger.info(f"Resultados de búsqueda (primeros 3 de {len(search_results)}):")
        for i, result in enumerate(search_results[:3]):
            logger.info(f"Resultado {i+1}: distancia={result.get('distance', 'N/A')}, filename={result.get('filename', 'N/A')}")
            if 'content' in result:
                logger.info(f"   Primeros 100 caracteres: {result['content'][:100]}...")
            elif 'text' in result:
                logger.info(f"   Primeros 100 caracteres: {result['text'][:100]}...")
        
        # Filtrar resultados con una distancia demasiado grande (poco relevantes)
        # Usar un umbral mucho más permisivo para capturar más contenido potencialmente relevante
        # Ajustar para el modelo de embeddings utilizado - las distancias son ahora mayores
        max_distance_threshold = float(os.getenv('MAX_DISTANCE_THRESHOLD', '20.0'))  # Aumentado desde 3.0 a 20.0
        filtered_results = [r for r in search_results if r.get('distance', 1.0) < max_distance_threshold]
        
        logger.info(f"Encontrados {len(filtered_results)} chunks con distancia < {max_distance_threshold} (de {len(search_results)} búsquedas)")
        
        # Ordenar los resultados filtrados por distancia
        filtered_results.sort(key=lambda x: x.get('distance', 1.0))
        
        # Tomar solo los k más relevantes después del filtrado
        return filtered_results[:k]
        
    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        """
        Genera una respuesta utilizando el modelo LLM aplicando técnicas de prompt engineering de Mistral AI.
        
        Args:
            query (str): Consulta del usuario
            context (str): Contexto relevante para responder
            sources (List[str], optional): Fuentes del contexto
            
        Returns:
            str: Respuesta generada
        """
        try:
            # Detectar si es un saludo simple
            greeting_words = ['hola', 'buenas', 'buen día', 'buen dia', 'buenos días', 'buenos dias', 
                             'buenas tardes', 'buenas noches', 'saludos', 'que tal', 'qué tal', 'como va', 'cómo va']
            
            # Lista de emojis para enriquecer las respuestas
            information_emojis = ["📚", "📖", "ℹ️", "📊", "🔍", "📝", "📋", "📈", "📌", "🧠"]
            greeting_emojis = ["👋", "😊", "🤓", "👨‍⚕️", "👩‍⚕️", "🎓", "🌟"]
            
            # Seleccionar emojis de manera pseudo-aleatoria pero consistente
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            info_emoji = information_emojis[query_hash % len(information_emojis)]
            greeting_emoji = greeting_emojis[query_hash % len(greeting_emojis)]
            
            # Detectar si hay saludo en la consulta
            has_greeting = any(word in query.lower().split() for word in greeting_words)
            
            is_greeting_only = query.lower().strip() in greeting_words or any(
                query.lower().strip() == word or query.lower().strip().startswith(word + " ")
                for word in greeting_words
            )
            
            # Si es solo un saludo, responder sin información adicional
            if is_greeting_only:
                greeting_responses = [
                    f"👨‍⚕️ ¡Hola! Soy DrCecim, tu asistente académico. ¿En qué puedo ayudarte hoy?",
                    f"👨‍⚕️ Saludos, soy DrCecim de la Facultad de Medicina. ¿Necesitas información sobre algo específico?",
                    f"👨‍⚕️ ¡Buenas! Soy DrCecim, ¿en qué puedo asistirte hoy?",
                    f"👨‍⚕️ Hola, soy DrCecim. ¿En qué puedo orientarte hoy?",
                    f"👨‍⚕️ Saludos. Soy DrCecim, tu asesor académico. ¿Necesitas ayuda con algún tema en particular?"
                ]
                import random
                return random.choice(greeting_responses)
            
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
            
            # Determinar saludos y prefijos en función de si hay saludo en la consulta
            greeting_used = None
            if has_greeting:
                for word in greeting_words:
                    if word in query.lower().split():
                        greeting_used = word
                        break
            
            # Determinar el saludo específico a usar en la respuesta
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
                else:
                    greeting_prefix = f"👨‍⚕️ ¡Hola! Soy DrCecim. "
            else:
                greeting_prefix = f"👨‍⚕️ "
            
            # Instrucción específica para el saludo
            greeting_instruction = ""
            if has_greeting:
                greeting_instruction = f"Inicia tu respuesta con '{greeting_prefix}'"
            else:
                greeting_instruction = f"Inicia tu respuesta con '{greeting_prefix}' sin mencionar el nombre DrCecim"
            
            # Aplicar técnicas de prompt engineering de Mistral AI
            system_prompt = f"""Eres un asistente virtual especializado, llamado DrCecim, de la Facultad de Medicina de la Universidad de Buenos Aires.

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

Cualquier información adicional debe ir en párrafos separados, nunca en la misma línea del ítem.

### Ejemplos INCORRECTOS (NUNCA USAR):
- Apercibimiento ⚠️ que se aplica en casos leves
- Suspensión ⚠️ para casos más graves

### Comportamiento con saludos:
Si el usuario te saluda con palabras como "hola", "buenos días", etc., responde con:
"👨‍⚕️ ¡Hola! Soy DrCecim, tu asistente de la Facultad de Medicina. ¿En qué puedo ayudarte hoy?"

Si el usuario NO te saluda, NUNCA menciones tu nombre "DrCecim" y comienza directamente con:
"👨‍⚕️ " seguido de la información solicitada.

### Instrucciones específicas:
{greeting_instruction}
Si la consulta NO incluye un saludo, NO añadas un saludo a tu respuesta.
Al hablar de tus capacidades, usa "Puedo ayudarte con..." o "Me puedes consultar sobre..." (NUNCA "Te puedo consultar").
SIEMPRE menciona "Facultad de Medicina" y no solo "Universidad de Buenos Aires".
NO INVENTES preguntas y respuestas adicionales que no son parte de la consulta original.
RESPONDE ÚNICAMENTE a la consulta del usuario, sin agregar información no solicitada.
Mantén tus respuestas BREVES y DIRECTAS.

<contexto>
{context}
</contexto>

<consulta>
{query}
</consulta>

Responde ÚNICAMENTE con información del contexto. NUNCA inventes información que no esté presente en el contexto."""
            
            user_prompt = f"Responde de manera directa y concisa a la consulta. Si necesitas hacer una lista, usa EXACTAMENTE el formato indicado en las instrucciones."
            
            # Formato completo del prompt
            prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.model.generate(prompt, max_length=512, temperature=0.7)
            
            # Post-procesar respuesta para asegurar el formato correcto
            response = re.sub(r'(Según el contexto|Basado en el contexto|De acuerdo con el contexto|Como se menciona en el contexto|Respuesta:|Según la información proporcionada)', '', response, flags=re.IGNORECASE)
            response = re.sub(r'<[^>]*>', '', response)  # Eliminar etiquetas HTML
            response = response.strip()
            
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
            
            # Si no es un saludo, eliminar cualquier mención a DrCecim
            if not has_greeting:
                response = re.sub(r'(?i)(Soy DrCecim\.?|DrCecim aquí\.?|DrCecim:)\s*', '', response)
                # También eliminar saludos si no hubo saludo en la consulta
                response = re.sub(r'(?i)(¡Hola!|Hola,|Buenos días|Buenas tardes|Buenas noches|Saludos)\s*', '', response)
            
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
        Procesa una consulta utilizando RAG para generar una respuesta.
        
        Args:
            query (str): Consulta del usuario
            num_chunks (int, optional): Número de fragmentos a recuperar
            
        Returns:
            Dict: Diccionario con la respuesta y detalles
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
            
            # Detectar si contiene un saludo
            greeting_words = ['hola', 'buenas', 'buen día', 'buen dia', 'buenos días', 'buenos dias', 
                             'buenas tardes', 'buenas noches', 'saludos', 'que tal', 'qué tal', 'como va', 'cómo va']
            
            # Identificar si hay un saludo en la consulta
            has_greeting = False
            greeting_used = None
            for word in greeting_words:
                if word in query.lower().split():
                    has_greeting = True
                    greeting_used = word
                    break
            
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
                else:
                    greeting_prefix = f"👨‍⚕️ ¡Hola! Soy DrCecim. "
            else:
                greeting_prefix = f"👨‍⚕️ "
            
            # Si es solo un saludo, responder directamente sin buscar embeddings
            if is_greeting_only:
                greeting_responses = [
                    f"👨‍⚕️ ¡Hola! Soy DrCecim, tu asistente académico. ¿En qué puedo ayudarte hoy?",
                    f"👨‍⚕️ Saludos, soy DrCecim de la Facultad de Medicina. ¿Necesitas información sobre algo específico?",
                    f"👨‍⚕️ ¡Buenas! Soy DrCecim, ¿en qué puedo asistirte hoy?",
                    f"👨‍⚕️ Hola, soy DrCecim. ¿En qué puedo orientarte hoy?",
                    f"👨‍⚕️ Saludos. Soy DrCecim, tu asesor académico. ¿Necesitas ayuda con algún tema en particular?"
                ]
                import random
                return {
                    "query": query,
                    "response": random.choice(greeting_responses),
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Verificar si es una consulta sobre las capacidades del bot
            meta_queries = [
                "qué hace", "que hace", "qué podés hacer", "que podes hacer",
                "en qué me podés ayudar", "en que me podes ayudar",
                "cómo me podés ayudar", "como me podes ayudar",
                "qué información tenés", "que informacion tenes",
                "qué información conocés", "que informacion conoces",
                "qué sabés", "que sabes", "para qué servís", "para que servis"
            ]
            is_meta_query = any(phrase in query.lower() for phrase in meta_queries)
            
            # Si es una pregunta sobre las capacidades del bot, no usar embeddings
            if is_meta_query:
                # Respuesta personalizada sobre capacidades
                if has_greeting:
                    meta_response = f"{greeting_prefix}Soy un asistente virtual especializado en información académica de la Facultad de Medicina de la Universidad de Buenos Aires. Puedo ayudarte con consultas sobre:\n- Condiciones de regularidad en la Facultad de Medicina 📚\n- Trámites administrativos para estudiantes de medicina 📋\n- Fechas importantes del calendario académico 🗓️\n- Requisitos de las materias y plan de estudios 📌\n- Información sobre el régimen disciplinario 🎓\n- Procedimientos de readmisión y sanciones 📄"
                else:
                    meta_response = f"👨‍⚕️ Puedo ayudarte con consultas sobre:\n- Condiciones de regularidad en la Facultad de Medicina 📚\n- Trámites administrativos para estudiantes de medicina 📋\n- Fechas importantes del calendario académico 🗓️\n- Requisitos de las materias y plan de estudios 📌\n- Información sobre el régimen disciplinario 🎓\n- Procedimientos de readmisión y sanciones 📄"
                
                return {
                    "query": query,
                    "response": meta_response,
                    "relevant_chunks": [],
                    "sources": []
                }
                
            # Para preguntas normales, seguir el flujo habitual de RAG
            if num_chunks is None:
                num_chunks = int(os.getenv('RAG_NUM_CHUNKS', 3))
            
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
                
                if "content" in chunk and chunk["content"].strip():
                    content = chunk["content"]
                    chunk_has_content = True
                elif "text" in chunk and chunk["text"].strip():
                    content = chunk["text"]
                    chunk_has_content = True
                else:
                    logger.warning(f"Chunk sin contenido válido: {chunk}")
                    continue  # Saltar este chunk
                
                # Evaluar si el chunk es realmente relevante para la consulta
                if 'distance' in chunk:
                    distance = chunk['distance']
                    logger.info(f"Chunk {i+1} distancia: {distance}")
                    if distance < relevance_threshold:
                        has_relevant_content = True
                        logger.info(f"Chunk {i+1} es relevante (distancia: {distance})")
                
                # Obtener información de la fuente
                source = ""
                if "filename" in chunk and chunk["filename"]:
                    source = chunk["filename"]
                    # Extraer solo el nombre del archivo sin la ruta
                    source = os.path.basename(source)
                    # Eliminar extensión .pdf
                    source = source.replace('.pdf', '')
                    if source and source not in sources:
                        sources.append(source)
                
                # Solo agregar chunks con contenido válido
                if chunk_has_content:
                    # Formatear el fragmento con su fuente pero sin usar FRAGMENTO en el mensaje
                    # para evitar que el modelo lo copie en la respuesta
                    formatted_chunk = f"Información de {source}:\n{content}"
                    context_chunks.append(formatted_chunk)
                    logger.info(f"Agregado chunk relevante de {source} (distancia: {chunk.get('distance', 'N/A')})")
            
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
                response = f"{greeting_prefix}{response}"
            elif not has_greeting and "DrCecim" in response:
                # Si no hay saludo, eliminar menciones a DrCecim
                response = re.sub(r'(?i)(Soy DrCecim\.?|DrCecim aquí\.?|DrCecim:)\s*', f'👨‍⚕️ ', response)
            
            if not any(emoji in response for emoji in information_emojis + greeting_emojis):
                response = f"{response} {info_emoji}"
            
            return {
                "query": query,
                "response": response,
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