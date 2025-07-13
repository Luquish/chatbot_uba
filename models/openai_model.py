"""
Implementación de modelos usando OpenAI.
"""
import logging
import numpy as np
from typing import List, Dict, Any

import openai

from models.base_model import BaseModel
from config.settings import API_TIMEOUT, MAX_OUTPUT_TOKENS, TEMPERATURE, TOP_P

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    """
    Clase para usar la API de OpenAI como modelo de lenguaje.
    """
    
    def __init__(self, model_name: str, api_key: str, timeout: int = API_TIMEOUT, max_output_tokens: int = MAX_OUTPUT_TOKENS):
        """
        Inicializa el cliente para OpenAI.
        
        Args:
            model_name (str): Nombre del modelo (ej: gpt-4o-mini)
            api_key (str): API key de OpenAI
            timeout (int): Timeout para las llamadas a la API
            max_output_tokens (int): Número máximo de tokens en la respuesta
        """
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
        self.max_output_tokens = max_output_tokens
        
    def generate(self, prompt: str, temperature: float = TEMPERATURE, top_p: float = TOP_P) -> str:
        """
        Genera texto usando un modelo de OpenAI.
        
        Args:
            prompt (str): Texto de entrada para el modelo
            temperature (float): Controla la aleatoriedad (0-1)
            top_p (float): Controla la diversidad (0-1)
            
        Returns:
            str: Texto generado
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_output_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error al llamar a OpenAI: {str(e)}")
            raise


class OpenAIEmbedding:
    """Servicio de generación de embeddings usando OpenAI."""

    def __init__(self, model_name: str, api_key: str, timeout: int = API_TIMEOUT):
        """Inicializa el cliente de OpenAI para embeddings.

        Args:
            model_name (str): Nombre del modelo (p.ej. ``text-embedding-3-small``).
            api_key (str): API key de OpenAI.
            timeout (int): Tiempo máximo de espera para las llamadas a la API.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

        # Configurar cliente
        openai.api_key = api_key

        logger.info(f"Modelo de embeddings OpenAI inicializado: {model_name}")

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Genera embeddings para una lista de textos.

        Args:
            texts (List[str]): Lista de textos a procesar.
            **kwargs: Argumentos opcionales para el procesamiento.

        Returns:
            np.ndarray: Matriz con los embeddings generados.
        """
        convert_to_numpy = kwargs.get("convert_to_numpy", True)
        normalize_embeddings = kwargs.get("normalize_embeddings", False)

        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format="float",
                timeout=self.timeout,
            )

            embeddings = [item.embedding for item in response.data]
            embeddings_array = np.array(embeddings, dtype=np.float32)

            if normalize_embeddings:
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                embeddings_array = embeddings_array / norms

            return embeddings_array if convert_to_numpy else embeddings

        except Exception as e:
            logger.error(
                f"Error al generar embeddings con OpenAI ({self.model_name}): {str(e)}"
            )
            raise

    def get_sentence_embedding_dimension(self) -> int:
        """Retorna la dimensión de los embeddings del modelo."""

        if "small" in self.model_name:
            return 1536
        elif "large" in self.model_name:
            return 3072
        else:
            return 1536