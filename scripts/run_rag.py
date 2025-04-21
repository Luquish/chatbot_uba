import os
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import torch
import faiss
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import psutil
import re
import random
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity
import openai
import time
import uuid

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
            "qué derechos tengo para inscribirme",
            # Nuevos ejemplos sobre denuncias y trámites administrativos
            "cómo presento una denuncia",
            "qué tengo que hacer para presentar una denuncia",
            "dónde puedo hacer una queja formal",
            "procedimiento para reportar un problema",
            "cómo puedo denunciar una situación irregular",
            "pasos para hacer una denuncia",
            "dónde se presentan las quejas",
            "quiero denunciar a alguien, qué hago",
            "cómo inicio un reclamo formal",
            "quiero reportar una irregularidad",
            "cómo suspender temporalmente mi condición de alumno",
            "puedo pedir suspensión de mis estudios",
            "qué pasa si pierdo la regularidad",
            "cómo solicito readmisión"
        ],
        'context': "El usuario necesita información sobre trámites administrativos, condiciones de regularidad o procedimientos formales"
    },
    'consulta_academica': {
        'examples': [
            "cuándo es el parcial",
            "dónde encuentro el programa",
            "cómo es la cursada",
            "qué necesito para aprobar",
            "cómo se evalúa la calidad de enseñanza",
            "qué materias puedo cursar",
            # Nuevos ejemplos relacionados con cuestiones académicas
            "cuántas materias tengo que aprobar para mantener regularidad",
            "qué pasa si tengo muchos aplazos",
            "cuántos aplazos puedo tener como máximo",
            "cuál es el porcentaje máximo de aplazos permitido",
            "en cuánto tiempo tengo que terminar la carrera",
            "plazo máximo para completar mis estudios",
            "cómo saber si soy alumno regular",
            "qué derechos tengo como alumno",
            "qué pasa si no apruebo suficientes materias",
            "cómo puedo cursar materias en otra facultad"
        ],
        'context': "El usuario necesita información académica sobre cursada, evaluación y aprobación"
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
            "qué pasa si rompo las reglas",
            # Nuevos ejemplos relacionados con el régimen disciplinario
            "qué sanciones hay si agredo a un profesor",
            "qué pasa si me comporto mal en la facultad",
            "cuáles son las sanciones disciplinarias",
            "quién puede denunciar una falta disciplinaria",
            "cómo es el proceso de un sumario disciplinario",
            "qué pasa si me suspenden preventivamente",
            "puedo apelar una sanción",
            "por cuánto tiempo pueden suspenderme",
            "qué pasa si falsifiqué un documento",
            "qué sucede si agravo a otro estudiante",
            "cuánto dura la suspensión por falta de respeto",
            "qué es un apercibimiento",
            "puedo estudiar en otra facultad si me suspenden",
            "cómo se presenta una denuncia por conducta inapropiada",
            "qué ocurre si adulteré un acta de examen"
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
    # Nuevas expansiones
    'denuncia': ['denuncia', 'queja', 'reclamo', 'reportar', 'irregularidad', 'problema', 'presentar', 'acusar'],
    'procedimiento': ['procedimiento', 'proceso', 'pasos', 'cómo', 'manera', 'forma', 'metodología', 'trámite'],
    'sancion': ['sanción', 'sanciones', 'castigo', 'penalidad', 'disciplina', 'apercibimiento', 'suspensión'],
    'sumario': ['sumario', 'investigación', 'proceso disciplinario', 'expediente'],
    'readmision': ['readmisión', 'readmitir', 'volver', 'reincorporación', 'reintegro'],
    'aprobacion': ['aprobar', 'aprobación', 'pasar materias', 'materias aprobadas', 'requisitos'],
    'suspension': ['suspensión', 'suspender', 'interrumpir', 'detener estudios', 'temporalmente']
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

# Clase base abstracta para búsquedas vectoriales
class VectorStore:
    def search(self, query_embedding: List[float], k: int) -> List[Dict]:
        """Búsqueda de vectores similares"""
        raise NotImplementedError("Este método debe ser implementado por las subclases")

# Implementación para FAISS
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

def load_model_with_fallback(model_path: str, load_kwargs: Dict) -> tuple:
    """
    Intenta cargar un modelo y, si falla, usa un modelo alternativo.
    
    Args:
        model_path (str): Nombre del modelo de OpenAI preferido
        load_kwargs (Dict): Argumentos para la configuración
        
    Returns:
        tuple: (modelo, nombre_del_modelo_cargado)
    """
    # Modelo de respaldo
    fallback_model = os.getenv('FALLBACK_MODEL_NAME', 'gpt-4.1-nano')
    
    try:
        logger.info(f"Intentando inicializar modelo OpenAI: {model_path}")
        model = OpenAIModel(
            model_name=model_path,
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=int(os.getenv('API_TIMEOUT', '30')),
            max_output_tokens=int(os.getenv('MAX_OUTPUT_TOKENS', '300'))
        )
        
        # Verificar que funciona
        test_prompt = "Escribe una palabra."
        model.generate(test_prompt, max_tokens=10)
        logger.info(f"Modelo OpenAI inicializado correctamente: {model_path}")
        return model, model_path
        
    except Exception as e:
        logger.warning(f"Error al inicializar modelo principal {model_path}: {str(e)}")
        logger.info(f"Intentando con modelo de fallback: {fallback_model}")
        
        try:
            # Intentar con el modelo de fallback
            model = OpenAIModel(
                model_name=fallback_model,
                api_key=os.getenv('OPENAI_API_KEY'),
                timeout=int(os.getenv('API_TIMEOUT', '30')),
                max_output_tokens=int(os.getenv('MAX_OUTPUT_TOKENS', '300'))
            )
            
            # Verificar que funciona
            test_prompt = "Escribe una palabra."
            model.generate(test_prompt, max_tokens=10)
            logger.info(f"Modelo de fallback OpenAI inicializado correctamente: {fallback_model}")
            return model, fallback_model
            
        except Exception as e2:
            raise RuntimeError(f"No se pudo inicializar ningún modelo de OpenAI: {str(e2)}")

# Clase base para modelos
class BaseModel:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

# Clase para usar la API de OpenAI
class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str, timeout: int = 30, max_output_tokens: int = 300):
        """
        Inicializa el cliente para OpenAI.
        
        Args:
            model_name (str): Nombre del modelo (ej: gpt-4o-mini)
            api_key (str): API key de OpenAI
            timeout (int): Timeout para las llamadas a la API
            max_output_tokens (int): Límite máximo de tokens para la respuesta
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        
        # Cargar parámetros de generación desde variables de entorno
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.top_p = float(os.getenv('TOP_P', '0.9'))
        self.top_k = int(os.getenv('TOP_K', '50'))
        
        # Configurar cliente de OpenAI
        openai.api_key = api_key
        
        logger.info(f"Modelo OpenAI inicializado: {model_name}")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Genera texto usando la API de OpenAI (Chat Completions).
        
        Args:
            prompt (str): Texto de entrada
            kwargs: Argumentos adicionales para la generación
            
        Returns:
            str: Texto generado
        """
        try:
            # Usar parámetros de kwargs si se proporcionan, si no usar los valores por defecto de las variables de entorno
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_output_tokens)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Crear los mensajes para el formato de chat de OpenAI
            messages = [{"role": "user", "content": prompt}]
            
            # Realizar la llamada a la API
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                timeout=self.timeout
            )
            
            # Extraer y retornar la respuesta
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error al generar texto con OpenAI ({self.model_name}): {str(e)}")
            raise

# Clase para crear embeddings usando OpenAI
class OpenAIEmbedding:
    def __init__(self, model_name: str, api_key: str, timeout: int = 30):
        """
        Inicializa el cliente para embeddings de OpenAI.
        
        Args:
            model_name (str): Nombre del modelo (ej: text-embedding-3-small)
            api_key (str): API key de OpenAI
            timeout (int): Timeout para las llamadas a la API
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        
        # Configurar cliente de OpenAI
        openai.api_key = api_key
        
        logger.info(f"Modelo de embeddings OpenAI inicializado: {model_name}")
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts (List[str]): Lista de textos para generar embeddings
            kwargs: Argumentos adicionales
            
        Returns:
            np.ndarray: Matriz de embeddings (una fila por texto)
        """
        try:
            # Crear embeddings usando la API de OpenAI
            response = openai.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float",
                timeout=self.timeout
            )
            
            # Extraer los embeddings de la respuesta
            embeddings = [item.embedding for item in response.data]
            
            # Convertir a numpy array
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error al generar embeddings con OpenAI ({self.model_name}): {str(e)}")
            raise
    
    def get_sentence_embedding_dimension(self) -> int:
        """
        Retorna la dimensión de los embeddings para compatibilidad con SentenceTransformer.
        
        Returns:
            int: Dimensión de los embeddings
        """
        # text-embedding-3-small tiene dimensión 1536
        if "small" in self.model_name:
            return 1536
        # text-embedding-3-large tiene dimensión 3072
        elif "large" in self.model_name:
            return 3072
        # Default para text-embedding-ada-002
        else:
            return 1536

class RAGSystem:
    def __init__(
        self,
        model_path: str = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
        embeddings_dir: str = os.getenv('EMBEDDINGS_DIR', 'data/embeddings'),
    ):
        """
        Inicializa el sistema RAG con OpenAI:
        - Usa GPT-4o mini de OpenAI como modelo principal
        - Si falla, usa GPT-4.1 nano como fallback
        - Usa text-embedding-3-small para embeddings
        """
        self.embeddings_dir = Path(embeddings_dir)
        
        # Histórico de conversaciones
        self.conversation_histories = {}  # Diccionario: user_id -> historial
        self.max_history_length = 5
        self.user_history = {}
        
        # Obtener configuración de API y llaves
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("Se requiere OPENAI_API_KEY para usar el sistema")
        
        # Modelos OpenAI
        self.primary_model_name = model_path
        self.fallback_model_name = os.getenv('FALLBACK_MODEL', 'gpt-4.1-nano')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        
        # Límite de tokens para respuestas (para controlar costos)
        self.max_output_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', '300'))
        
        # Timeout para API
        self.api_timeout = int(os.getenv('API_TIMEOUT', '30'))
        
        # Normalizar los ejemplos de intenciones para mejorar la detección
        self.normalized_intent_examples = self._normalize_intent_examples()
        logger.info("Ejemplos de intenciones normalizados para mejorar la clasificación")
        
        # Inicializar modelos de OpenAI
        try:
            logger.info(f"Inicializando modelo principal OpenAI: {self.primary_model_name}")
            self.model = self._initialize_openai_model(self.primary_model_name)
            
            # Inicializar modelo de embeddings de OpenAI
            logger.info(f"Inicializando modelo de embeddings OpenAI: {self.embedding_model_name}")
            self.embedding_model = OpenAIEmbedding(
                model_name=self.embedding_model_name,
                api_key=self.openai_api_key,
                timeout=self.api_timeout
            )
            logger.info(f"Dimensión del modelo de embeddings: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Error al inicializar modelo OpenAI: {str(e)}")
            raise RuntimeError(f"No se pudo inicializar el modelo OpenAI: {str(e)}")
        
        # Inicializar el almacén vectorial
        self.vector_store = self._initialize_vector_store()

        # Configurar umbral de similitud
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))

    def _initialize_openai_model(self, model_name: str) -> OpenAIModel:
        """
        Inicializa el modelo de OpenAI con fallback.
        
        Args:
            model_name (str): Nombre del modelo principal
            
        Returns:
            OpenAIModel: Modelo inicializado
        """
        try:
            # Intentar inicializar el modelo principal
            model = OpenAIModel(
                model_name=model_name,
                api_key=self.openai_api_key,
                timeout=self.api_timeout,
                max_output_tokens=self.max_output_tokens
            )
            
            # Verificar que funciona
            test_prompt = "Escribe una palabra."
            model.generate(test_prompt, max_tokens=10)
            logger.info(f"Modelo OpenAI inicializado correctamente: {model_name}")
            return model
            
        except Exception as e:
            logger.warning(f"Error al inicializar modelo principal {model_name}: {str(e)}")
            logger.info(f"Intentando con modelo de fallback: {self.fallback_model_name}")
            
            try:
                # Intentar con el modelo de fallback
                model = OpenAIModel(
                    model_name=self.fallback_model_name,
                    api_key=self.openai_api_key,
                    timeout=self.api_timeout,
                    max_output_tokens=self.max_output_tokens
                )
                
                # Verificar que funciona
                test_prompt = "Escribe una palabra."
                model.generate(test_prompt, max_tokens=10)
                logger.info(f"Modelo de fallback OpenAI inicializado correctamente: {self.fallback_model_name}")
                return model
                
            except Exception as e2:
                raise RuntimeError(f"No se pudo inicializar ningún modelo de OpenAI: {str(e2)}")


    def _initialize_vector_store(self) -> VectorStore:
        """
        Inicializa el almacén vectorial usando FAISS.
        
        Returns:
            VectorStore: Implementación del almacén vectorial FAISS
        """
        index_path = str(self.embeddings_dir / 'faiss_index.bin')
        metadata_path = str(self.embeddings_dir / 'metadata.csv')
        logger.info(f"Inicializando índice FAISS desde {index_path}")
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
            
            # Formato específico para modelo E5: Instruct + Query
            task_description = "Recuperar información relevante sobre procedimientos y reglamentos administrativos universitarios"
            formatted_query = f"Instruct: {task_description}\nQuery: {expanded_query}"
            logger.info(f"Consulta formateada para E5: {formatted_query}")
            
            query_embedding = self.embedding_model.encode(
                [formatted_query],
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
        Genera una respuesta usando el LLM definido en el archivo .env.
        Optimizado para GPT-4o mini para producir respuestas concisas y relevantes.
        """
        # Seleccionar un emoji aleatorio para la respuesta
        emoji = random.choice(information_emojis)
        
        # Versión completa de las FAQ
        faqs_complete = """
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
            sources_text = f"\nFUENTES CONSULTADAS:\n{', '.join(formatted_sources)}"

        # Prompt optimizado para OpenAI
        system_message = f"""Sos DrCecim, un asistente virtual especializado de la Facultad de Medicina UBA. Tu tarea es proporcionar respuestas son sobre administración y trámites de la facultad y deben ser breves, precisas y útiles.

SOBRE TI:
- Te llamas DrCecim y eres un asistente virtual de la Facultad de Medicina UBA
- Fuiste creado para ayudar a responder preguntas sobre trámites, reglamentos y procedimientos
- Cuando te pregunten sobre tu identidad, debes responder que eres DrCecim
- No confundas preguntas sobre tu identidad con preguntas sobre la identidad del usuario
- Cuando te pregunten como estas, o alguna relacionada a tu estado, debes responder que estas bien y listo para ayudar
- IMPORTANTE: Solo debes saludar en tu primera interacción con el usuario. En las siguientes respuestas, ve directo al punto
- Siempre debes responder en modo casual, como un amigo
- Siempre debes responder en modo informal, como un mensaje de WhatsApp

INFORMACIÓN RELEVANTE:
{context}

PREGUNTAS FRECUENTES:
{faqs_complete}

{sources_text}"""

        # Agregar el historial de la conversación al mensaje del usuario
        conversation_history = ""
        if hasattr(self, 'user_history') and self.user_history:
            conversation_history = "\nHISTORIAL DE LA CONVERSACIÓN:\n"
            for entry in list(self.user_history.values())[0][-3:]:  # Últimas 3 interacciones
                conversation_history += f"Usuario: {entry['query']}\nDrCecim: {entry['response']}\n"

        user_message = f"""CONSULTA ACTUAL: {query}

{conversation_history}

RESPONDE SIGUIENDO ESTAS REGLAS:
1. Sé muy conciso y directo
2. Usa la información de los documentos oficiales primero
3. Si hay documentos específicos, cita naturalmente su origen ("Según el reglamento...")
4. NO uses NUNCA formato Markdown (como asteriscos para negrita o cursiva) ya que esto no se procesa correctamente en WhatsApp
5. Para enfatizar texto, usa MAYÚSCULAS, comillas o asteriscos
6. Usa viñetas con guiones (-) cuando sea útil para mayor claridad
7. Si la información está incompleta, sugiere contactar a @cecim.nemed por instagram
8. No hagas preguntas adicionales
9. Si ya hubo un saludo previo en el historial, NO vuelvas a saludar"""

        # Llamada a la API de OpenAI con mensajes formatados
        try:
            # Obtener parámetros de generación de variables de entorno
            temperature = float(os.getenv('TEMPERATURE', '0.7'))
            top_p = float(os.getenv('TOP_P', '0.9'))
            
            response = openai.chat.completions.create(
                model=self.model.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_output_tokens,
                timeout=self.api_timeout
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error al generar respuesta con OpenAI: {str(e)}")
            return f"{emoji} Lo siento, hubo un error al generar la respuesta. Por favor, intenta de nuevo o contacta a @cecim.nemed por instagram."

        # Eliminar cualquier formato Markdown que pueda haberse colado
        response_text = re.sub(r'\*\*(.+?)\*\*', r'\1', response_text)  # Eliminar negrita
        response_text = re.sub(r'\*(.+?)\*', r'\1', response_text)  # Eliminar cursiva
        response_text = re.sub(r'\_\_(.+?)\_\_', r'\1', response_text)  # Eliminar subrayado
        response_text = re.sub(r'\_(.+?)\_', r'\1', response_text)  # Eliminar cursiva con guiones bajos

        # Asegurar que la respuesta tenga el emoji (si no comienza ya con uno)
        emoji_pattern = r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]'
        if not re.match(emoji_pattern, response_text.strip()[0:1]):
            response_text = f"{emoji} {response_text}"
        
        return response_text

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
        Procesa una consulta del usuario sin depender de detección de intenciones.
        La comprensión de la consulta se delega al modelo de lenguaje.
        
        Args:
            query (str): La consulta del usuario
            user_id (str, optional): ID del usuario (número de teléfono)
            user_name (str, optional): Nombre del usuario si está disponible
        """
        try:
            # Si no hay user_id, generar uno
            if user_id is None:
                user_id = str(uuid.uuid4())
            
            # Obtener historial de mensajes
            history = self.get_user_history(user_id)
            
            # Usar el historial para contextualizar la consulta si es necesario
            context_from_history = self._summarize_previous_message(user_id) if history else ""
            
            # Normalizar la consulta
            query_original = query
            query = query.lower().strip()
            query = unidecode(query)  # Eliminar tildes
            query = re.sub(r'[^\w\s]', '', query)  # Eliminar signos de puntuación
            query = re.sub(r'\s+', ' ', query).strip()  # Normalizar espacios
            
            if query_original != query:
                logger.info(f"Consulta normalizada: '{query_original}' → '{query}'")
            
            # Verificar si es una consulta muy corta o de saludo simple
            if len(query.split()) <= 2 and any(saludo in query for saludo in ['hola', 'buenas', 'saludos', 'hey']):
                # Respuesta de saludo simple
                return {
                    "query": query_original,
                    "response": f"{random.choice(greeting_emojis)} ¡Hola! Soy DrCecim, un asistente virtual de la Facultad de Medicina de la UBA. Estoy aquí para ayudarte con trámites y consultas. ¿En qué puedo asistirte hoy?",
                    "query_type": "saludo",
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Obtener chunks relevantes
            num_chunks = int(os.getenv('RAG_NUM_CHUNKS', 3))
            relevant_chunks = self.retrieve_relevant_chunks(query_original, k=num_chunks)
            
            # Si no hay resultados relevantes, intentar con la consulta normalizada
            if not relevant_chunks:
                relevant_chunks = self.retrieve_relevant_chunks(query, k=num_chunks)
            
            # Si aún no hay resultados, reducir el umbral de similitud temporalmente
            if not relevant_chunks:
                logger.info("Intentando búsqueda con umbral reducido...")
                original_threshold = self.similarity_threshold
                self.similarity_threshold = 0.1  # Reducir temporalmente el umbral
                relevant_chunks = self.retrieve_relevant_chunks(query_original, k=num_chunks)
                self.similarity_threshold = original_threshold  # Restaurar umbral
            
            # Construir contexto con los chunks relevantes
            context_chunks = []
            sources = []
            
            for chunk in relevant_chunks:
                # Extraer el contenido del chunk
                if "content" in chunk and chunk["content"].strip():
                    content = chunk["content"]
                elif "text" in chunk and chunk["text"].strip():
                    content = chunk["text"]
                else:
                    continue
                
                # Extraer la fuente
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
            
            # Si no hay contexto suficiente, dar una respuesta genérica
            if not context.strip():
                logger.warning("No se encontró contexto suficientemente relevante")
                emoji = random.choice(information_emojis)
                standard_no_info_response = f"{emoji} Lo siento, no encontré información específica sobre esta consulta en mis documentos. Te sugiero escribir a **alumnos@fmed.uba.ar** para obtener la información precisa que necesitas."
                
                return {
                    "query": query_original,
                    "response": standard_no_info_response,
                    "relevant_chunks": [],
                    "sources": [],
                    "query_type": "sin_información"
                }
            
            logger.info(f"Se encontraron {len(context_chunks)} fragmentos relevantes de {len(sources)} fuentes")
            
            # Generar respuesta
            response = self.generate_response(query_original, context, sources)
            
            # Actualizar historial del usuario
            if user_id:
                self.update_user_history(user_id, query_original, response)
            
            # Devolver respuesta
            return {
                "query": query_original,
                "response": response,
                "relevant_chunks": relevant_chunks,
                "sources": sources,
                "query_type": "consulta_general"
            }
            
        except Exception as e:
            logger.error(f"Error en process_query: {str(e)}", exc_info=True)
            emoji = random.choice(warning_emojis)
            error_response = f"{emoji} Lo siento, ocurrió un problema al procesar tu consulta. Por favor, inténtalo de nuevo o contacta a **alumnos@fmed.uba.ar** si el problema persiste."
            
            return {
                "query": query,
                "response": error_response,
                "error": str(e),
                "query_type": "error"
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
        
        # Crear las respuestas detalladas para cada FAQ
        faq_responses = {
            "constancia": """📋 Constancia de alumno regular:
Puedes tramitar la constancia de alumno regular en el Sitio de Inscripciones siguiendo estos pasos:
- Paso 1: Ingresar tu DNI y contraseña.
- Paso 2: Seleccionar la opción "Constancia de alumno regular" en el inicio de trámites.
- Paso 3: Imprimir la constancia. Luego, deberás presentarte con tu Libreta Universitaria o DNI y el formulario impreso (1 hoja que posee 3 certificados de alumno regular) en la ventanilla del Ciclo Biomédico.""",
            
            "baja": """📝 Baja de materia:
El tiempo máximo para dar de baja una materia es:
- 2 semanas antes del primer parcial, o
- Hasta el 25% de la cursada en asignaturas sin examen parcial.

Para dar de baja una materia, sigue estos pasos en el Sitio de Inscripciones:
- Paso 1: Ingresar tu DNI y contraseña.
- Paso 2: Seleccionar "Baja de asignatura".
- Paso 3: Imprimir el certificado de baja.

Una vez finalizado el trámite, el estado será "Resuelto Positivamente" y no deberás acudir a la Dirección de Alumnos.""",
            
            "anulacion": """❌ Anulación de inscripción a final:
Para anular la inscripción a un final, debes acudir a la ventanilla del Ciclo Biomédico presentando el número de constancia generado durante el trámite de inscripción.""",
            
            "inscripcion": """📊 No lograr inscripción o asignación a materia:
Si no logras inscribirte o no te asignan una materia, debes dirigirte a la cátedra o departamento correspondiente y solicitar la inclusión en lista, presentando tu Libreta Universitaria o DNI.""",
            
            "reincorporacion": """🔄 Reincorporación:
La reincorporación se solicita a través del Sitio de Inscripciones, seleccionando la opción "Reincorporación a la carrera":
- Para la 1ª reincorporación: El trámite es automático y aparece resuelto positivamente en el sistema, sin necesidad de trámite en ventanilla.
- Si ya fuiste reincorporado anteriormente: Debes realizar el trámite, imprimirlo (consta de 2 hojas: 1 certificado y 1 constancia) y presentarlo en la ventanilla del Ciclo Biomédico, donde la Comisión de Readmisión resolverá tu caso.""",
            
            "recursada": """🔁 Recursada (inscripción por segunda vez):
Para solicitar una recursada, genera el trámite en el Sitio de Inscripciones siguiendo estos pasos:
- Paso 1: Ingresar tu DNI y contraseña.
- Paso 2: Seleccionar "Recursada".

El trámite es automático y, si aparece resuelto positivamente en el sistema, no necesitas acudir a ventanilla.
- Si en el sistema apareces como dado DE BAJA en la cursada anterior, solo debes generar el trámite y te inscribirás como la primera vez, sin abonar arancel.
- Si no apareces dado DE BAJA, deberás:
  1. Realizar el trámite.
  2. Generar e imprimir el talón de pago.
  3. Pagar en la Dirección de Tesorería.
  4. Presentar un comprobante de pago en los buzones del Ciclo Biomédico.""",
            
            "tercera": """3️⃣ Tercera cursada:
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
  4. Presentar un comprobante de pago en el buzón del Ciclo Biomédico.""",
            
            "cuarta": """4️⃣ Cuarta cursada o más:
Para la cuarta cursada o más, genera el trámite en el Sitio de Inscripciones con los siguientes pasos:
- Paso 1: Dirigirte a Inscripciones.
- Paso 2: Ingresar tu DNI y contraseña.
- Paso 3: Seleccionar "4º Cursada o más".
- Paso 4: Imprimir la constancia y el certificado.

Luego, deberás presentarte con tu Libreta Universitaria y las constancias impresas en la ventanilla del Ciclo Biomédico y acudir a la Dirección de Alumnos.""",
            
            "prorroga": """⏳ Prórroga de materias:
Para solicitar la prórroga de una asignatura, sigue estos pasos en el Sitio de Inscripciones:
- Paso 1: Dirigirte a Inscripciones.
- Paso 2: Ingresar tu DNI y contraseña.
- Paso 3: Seleccionar "Prórroga de asignatura".
- Paso 4: Imprimir la constancia.

Si se trata de la primera o segunda prórroga, el trámite se resuelve positivamente. Si es la tercera o una prórroga superior, deberás presentar la constancia impresa junto con tu Libreta Universitaria en la ventanilla del Ciclo Biomédico."""
        }
        
        # Buscar coincidencias
        for faq_type, keywords in faq_keywords.items():
            if any(keyword in query_normalized for keyword in keywords):
                return faq_responses.get(faq_type, "")
        
        return None

    def _verify_response(self, response: str, context: str, intent: str) -> tuple:
        """
        Verifica la calidad de la respuesta generada con criterios mejorados
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
        if intent == 'consulta_reglamento':
            if not any(word in response.lower() for word in ['artículo', 'reglamento', 'normativa', 'sanción', 'según', 'establece']):
                verification_score *= 0.8
        elif intent == 'consulta_administrativa':
            if not any(word in response.lower() for word in ['trámite', 'pasos', 'procedimiento', 'debes', 'podrás', 'deberás']):
                verification_score *= 0.8
            
            # Nuevo: Verificar si hay términos específicos para denuncias
            if "denuncia" in context.lower() and not any(word in response.lower() for word in ['denuncia', 'reportar', 'presentar', 'escrito']):
                verification_score *= 0.7
        
        # Verificar presencia de elementos estructurales
        if not any(emoji in response for emoji in (greeting_emojis + information_emojis)):
            verification_score *= 0.9
        
        return response, verification_score

    def get_user_history(self, user_id: str) -> list:
        """Obtiene el historial de mensajes del usuario."""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        return self.user_history[user_id]

    def update_user_history(self, user_id: str, query: str, response: str):
        """Actualiza el historial de mensajes del usuario."""
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        # Agregar nuevo mensaje al historial
        self.user_history[user_id].append({
            "query": query,
            "response": response,
            "timestamp": time.time()
        })
        
        # Limitar a los últimos 5 mensajes
        self.user_history[user_id] = self.user_history[user_id][-5:]

def main():
    """Función principal para ejecutar el sistema RAG."""
    # Verificar que existe la API key de OpenAI
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Se requiere OPENAI_API_KEY para usar el sistema")
        return
        
    # Inicializar sistema RAG
    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"Error al inicializar el sistema: {str(e)}")
        return
    
    # Ejemplo de uso
    print("\nBienvenido al sistema RAG de DrCecim")
    print("Este sistema usa modelos de OpenAI para responder consultas sobre la Facultad de Medicina UBA")
    print("Escribe 'salir' para terminar")
    
    while True:
        query = input("\nIngrese su consulta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
            
        try:
            result = rag.process_query(query)
            print("\nRespuesta:", result['response'])
            if result.get('sources'):
                print("\nFuentes consultadas:")
                for source in result['sources']:
                    print(f"- {source}")
        except Exception as e:
            logger.error(f"Error al procesar la consulta: {str(e)}")
            print("Lo siento, hubo un error al procesar tu consulta.")

if __name__ == "__main__":
    main() 