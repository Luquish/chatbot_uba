"""
Implementaciones de almacenes vectoriales para la búsqueda semántica.
Sistema de vectores basado en PostgreSQL con pgvector para el chatbot UBA.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from config.settings import SIMILARITY_THRESHOLD
from services.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


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


class PostgreSQLVectorStore(VectorStore):
    """
    Almacén vectorial que utiliza PostgreSQL con pgvector para búsquedas eficientes.
    """
    
    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        """
        Inicializa el almacén vectorial PostgreSQL.
        
        Args:
            threshold (float): Umbral mínimo de similitud para incluir resultados
        """
        self.threshold = threshold
        self.vector_db_service = VectorDBService()
        
        # Verificar conexión al inicializar
        if self.vector_db_service.test_connection():
            logger.info("PostgreSQL Vector Store inicializado correctamente")
        else:
            raise RuntimeError("No se pudo establecer conexión con la base de datos")
    
    def search(self, query_embedding, k=5, threshold=None):
        """
        Busca los documentos más similares a la consulta utilizando PostgreSQL.
        
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
            
        # Asegurar que sea unidimensional para pgvector
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()
            
        # Realizar búsqueda de similitud
        results = self.vector_db_service.similarity_search(
            query_embedding=query_embedding,
            k=k,
            threshold=threshold
        )
        
        # Adaptar formato de respuesta para compatibilidad
        adapted_results = []
        for result in results:
            adapted_result = {
                'text': result.get('text_content', ''),
                'document_id': result.get('document_id', ''),
                'chunk_id': result.get('chunk_id', ''),
                'similarity': result.get('similarity', 0.0),
                'distance': result.get('distance', 1.0),
                'metadata': result.get('metadata', {})
            }
            
            # Añadir campos adicionales del metadata si existen
            if 'metadata' in result and result['metadata']:
                for key, value in result['metadata'].items():
                    if key not in adapted_result:
                        adapted_result[key] = value
            
            adapted_results.append(adapted_result)
                
        return adapted_results 

    def search_by_metadata(self, metadata_filter: Dict, limit: int = 5) -> List[Dict]:
        """
        Busca en el almacén por coincidencia exacta de metadatos.
        
        Args:
            metadata_filter (Dict): Diccionario con los filtros de metadatos
            limit (int): Número máximo de resultados a devolver
            
        Returns:
            List[Dict]: Lista de documentos que coinciden con los criterios
        """
        # Para implementación futura - búsqueda por metadatos específicos
        logger.warning("Búsqueda por metadatos no implementada aún en PostgreSQL Vector Store")
        return []
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas del almacén vectorial.
        
        Returns:
            Dict: Estadísticas del almacén
        """
        stats = self.vector_db_service.get_database_stats()
        
        # Adaptar formato para compatibilidad
        adapted_stats = {
            'total_vectors': stats.get('total_embeddings', 0),
            'vector_dimension': 1536,  # OpenAI text-embedding-3-small
            'metadata_records': stats.get('total_embeddings', 0),
            'unique_documents': stats.get('unique_documents', 0),
            'using_postgresql': True,
            'using_pgvector': True,
            'threshold': self.threshold,
            'recent_documents': stats.get('recent_documents', []),
            'table_info': stats.get('table_info', {})
        }
        
        return adapted_stats

    def store_embeddings(self, embeddings: np.ndarray, metadata_df: pd.DataFrame) -> bool:
        """
        Almacena embeddings en la base de datos PostgreSQL.
        
        Args:
            embeddings (np.ndarray): Array de embeddings
            metadata_df (pd.DataFrame): DataFrame con metadatos
            
        Returns:
            bool: True si se almacenaron exitosamente
        """
        return self.vector_db_service.store_embeddings(embeddings, metadata_df)

    def delete_document(self, document_id: str) -> bool:
        """
        Elimina todos los embeddings de un documento.
        
        Args:
            document_id (str): ID del documento a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente
        """
        return self.vector_db_service.delete_document_embeddings(document_id)

    def create_indices(self) -> bool:
        """
        Crea índices para optimizar las búsquedas.
        
        Returns:
            bool: True si se crearon exitosamente
        """
        return self.vector_db_service.create_index()


# Función de conveniencia para crear un almacén vectorial
def create_vector_store(threshold: float = SIMILARITY_THRESHOLD) -> PostgreSQLVectorStore:
    """
    Crea un almacén vectorial PostgreSQL con configuración automática.
    
    Args:
        threshold (float): Umbral de similitud
        
    Returns:
        PostgreSQLVectorStore: Almacén vectorial configurado
    """
    return PostgreSQLVectorStore(threshold=threshold)