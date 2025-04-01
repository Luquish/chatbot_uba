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
        return self.vector_store.search(query_embedding, k)
        
    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        """
        Genera una respuesta basada en la consulta y el contexto.
        
        Args:
            query (str): Consulta del usuario
            context (str): Contexto recuperado del RAG
            sources (List[str]): Lista de fuentes consultadas
            
        Returns:
            str: Respuesta generada
        """
        # Preparar la lista de fuentes para incluirla en el prompt
        sources_text = ""
        if sources and len(sources) > 0:
            sources_text = "FUENTES CONSULTADAS:\n" + "\n".join([f"- {source}" for source in sources])
        
        prompt = f"""Eres un asistente virtual de la Universidad de Buenos Aires (UBA) llamado DrCecim que ayuda a los alumnos con consultas acerca de la univerdad, no sobre medicina. 
        Tu función es responder preguntas sobre temas de la universidad basándote en información verificada.

INSTRUCCIONES:
1. Responde de manera clara, precisa y en español.
2. Si la pregunta no está relacionada con medicina, responde amablemente que estás especializado en temas médicos.
3. Utiliza un tono profesional pero amigable.
4. IMPORTANTE: Cita las fuentes de donde proviene la información en tu respuesta.
5. Si utilizas información de algún fragmento específico, menciona la fuente correspondiente.
6. No inventes información que no esté en el contexto proporcionado.
7. Si no tienes suficiente información, indícalo honestamente.
8. Evita dar consejos médicos personalizados - recuerda que no reemplazas a un médico.
9. Al final de tu respuesta, incluye una sección de "Referencias" con las fuentes utilizadas.

CONTEXTO:
{context}

{sources_text}

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""
        
        generation_kwargs = {
            "max_length": int(os.getenv('MAX_LENGTH', 512)),
            "temperature": float(os.getenv('TEMPERATURE', 0.7)),
            "top_p": float(os.getenv('TOP_P', 0.9)),
            "top_k": int(os.getenv('TOP_K', 50))
        }
        
        logger.info(f"Generando respuesta para consulta: {query[:50]}...")
        
        try:
            response_text = self.model.generate(prompt, **generation_kwargs)
            
            # Extraer solo la respuesta
            response_marker = "RESPUESTA:"
            response_start = response_text.rfind(response_marker)
            
            if response_start != -1:
                response = response_text[response_start + len(response_marker):].strip()
            else:
                response = response_text.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error al generar respuesta: {str(e)}")
            return f"Lo siento, hubo un error al generar la respuesta. Por favor, intenta nuevamente en unos momentos."
        
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
            if num_chunks is None:
                num_chunks = int(os.getenv('RAG_NUM_CHUNKS', 3))
            
            # Encontrar fragmentos relevantes
            relevant_chunks = self.retrieve_relevant_chunks(query, k=num_chunks)
            
            # Construir contexto de manera segura, incluyendo la fuente de cada fragmento
            context_chunks = []
            sources = []
            
            for i, chunk in enumerate(relevant_chunks):
                if "content" in chunk:
                    content = chunk["content"]
                elif "text" in chunk:
                    content = chunk["text"]
                else:
                    logger.warning(f"Chunk sin campo 'content' o 'text': {chunk}")
                    content = "No hay contenido disponible para este fragmento."
                
                # Obtener información de la fuente
                source = ""
                if "filename" in chunk:
                    source = chunk["filename"]
                    if source not in sources:
                        sources.append(source)
                
                # Formatear el fragmento con su fuente
                formatted_chunk = f"[FRAGMENTO {i+1}] (Fuente: {source})\n{content}"
                context_chunks.append(formatted_chunk)
            
            # Unir los chunks para formar el contexto
            context = '\n\n'.join(context_chunks)
            
            # Si no hay contexto, usar un mensaje informativo
            if not context.strip():
                context = "No se encontró información relevante en la base de conocimiento."
                logger.warning("No se encontró contexto relevante para la consulta.")
            
            # Generar respuesta
            response = self.generate_response(query, context, sources)
            
            # Crear objeto de respuesta completo
            return {
                "query": query,
                "response": response,
                "relevant_chunks": relevant_chunks,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error en process_query: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": "Lo siento, tuve un problema procesando tu consulta. Por favor, intenta de nuevo.",
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