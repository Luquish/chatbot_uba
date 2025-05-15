"""
Implementaciones de almacenes vectoriales para la búsqueda semántica.
"""
import os
import logging
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict

from config.settings import USE_GCS, GCS_BUCKET_NAME, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Importar la integración con Google Cloud Storage si está disponible
try:
    from scripts.gcs_storage import load_faiss_from_gcs
    GCS_AVAILABLE = True
    logger.info("Módulo de Google Cloud Storage disponible")
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("Módulo de Google Cloud Storage no disponible, usando sistema de archivos local")


class VectorStore:
    """
    Clase base abstracta para búsquedas vectoriales.
    """
    
    def search(self, query_embedding, k=5, threshold=None):
        """
        Busca los documentos más similares a la consulta.
        
        Args:
            query_embedding: Embedding de la consulta
            k (int): Número de resultados a devolver
            threshold (float): Umbral mínimo de similitud
            
        Returns:
            List[Dict]: Lista de documentos similares
        """
        raise NotImplementedError("Este método debe ser implementado por las subclases")


class FAISSVectorStore(VectorStore):
    """
    Almacén vectorial que utiliza FAISS para búsquedas eficientes.
    """
    
    def __init__(self, faiss_index_path: str, metadata_path: str, threshold: float = SIMILARITY_THRESHOLD):
        """
        Inicializa el almacén vectorial FAISS.
        
        Args:
            faiss_index_path (str): Ruta al archivo de índice FAISS
            metadata_path (str): Ruta al archivo CSV con metadatos
            threshold (float): Umbral mínimo de similitud para incluir resultados
        """
        self.threshold = threshold
        
        # Cargar desde GCS o sistema de archivos local
        if USE_GCS and GCS_AVAILABLE:
            logger.info(f"Cargando índice FAISS desde GCS bucket: {GCS_BUCKET_NAME}")
            self.index = load_faiss_from_gcs(GCS_BUCKET_NAME, os.path.basename(faiss_index_path))
            # Cargar metadatos
            metadata_gcs_path = os.path.join(GCS_BUCKET_NAME, os.path.basename(metadata_path))
            self.metadata_df = pd.read_csv(metadata_gcs_path)
        else:
            logger.info(f"Cargando índice FAISS desde archivo local: {faiss_index_path}")
            if os.path.exists(faiss_index_path):
                self.index = faiss.read_index(faiss_index_path)
            else:
                raise FileNotFoundError(f"Archivo de índice FAISS no encontrado: {faiss_index_path}")
            
            # Cargar metadatos
            if os.path.exists(metadata_path):
                self.metadata_df = pd.read_csv(metadata_path)
            else:
                raise FileNotFoundError(f"Archivo de metadatos no encontrado: {metadata_path}")
        
        logger.info(f"Índice FAISS cargado con {self.index.ntotal} vectores y dimensión {self.index.d}")
        logger.info(f"Metadatos cargados con {len(self.metadata_df)} registros")
        
    def search(self, query_embedding, k=5, threshold=None):
        """
        Busca los documentos más similares a la consulta utilizando FAISS.
        
        Args:
            query_embedding: Embedding de la consulta
            k (int): Número de resultados a devolver
            threshold (float): Umbral mínimo de similitud (opcional)
            
        Returns:
            List[Dict]: Lista de documentos similares con metadatos
        """
        if threshold is None:
            threshold = self.threshold
            
        # Asegurar que query_embedding sea un array NumPy
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
        # Reshape si es necesario para FAISS
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Asegurar que sea float32 (requerido por FAISS)
        query_embedding = query_embedding.astype(np.float32)
        
        # Buscar en el índice FAISS
        distances, indices = self.index.search(query_embedding, k=k)
        
        # Filtrar resultados usando el umbral de similitud
        results = []
        for i, idx in enumerate(indices[0]):
            # FAISS devuelve distancias (menor es mejor), convertir a similitud
            similarity = 1.0 - distances[0][i]
            
            if idx != -1 and similarity >= threshold:
                # Obtener metadatos del documento
                metadata = self.metadata_df.iloc[idx].to_dict()
                
                # Añadir similitud al resultado
                metadata['similarity'] = float(similarity)
                
                results.append(metadata)
                
        return results 