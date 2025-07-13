"""
Implementaciones de almacenes vectoriales para la búsqueda semántica.
Actualizado para usar automáticamente Google Cloud Storage.
"""
import os
import logging
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Optional

from config.settings import USE_GCS, GCS_BUCKET_NAME, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Importar la nueva integración con Google Cloud Storage
try:
    from scripts.gcs_storage import load_faiss_from_gcs
    GCS_AVAILABLE = True
    logger.info("Módulo de Google Cloud Storage disponible (legacy)")
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("Módulo de Google Cloud Storage legacy no disponible")

# Importar el nuevo servicio de GCS
try:
    # Importar desde el nuevo repo drcecim_upload si está disponible
    import sys
    import importlib.util
    
    # Intentar cargar el nuevo servicio GCS
    drcecim_upload_path = os.path.join(os.path.dirname(__file__), '..', '..', 'drcecim_upload')
    if os.path.exists(drcecim_upload_path):
        sys.path.insert(0, drcecim_upload_path)
        from services.gcs_service import GCSService, load_embeddings_from_gcs
        NEW_GCS_AVAILABLE = True
        logger.info("Nuevo servicio de Google Cloud Storage disponible")
    else:
        NEW_GCS_AVAILABLE = False
        logger.info("Nuevo servicio de GCS no disponible, usando sistema legacy")
        
except ImportError:
    NEW_GCS_AVAILABLE = False
    logger.warning("Nuevo servicio de Google Cloud Storage no disponible")


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
    Actualizado para usar automáticamente Google Cloud Storage.
    """
    
    def __init__(self, faiss_index_path: str, metadata_path: str, threshold: float = SIMILARITY_THRESHOLD, 
                 bucket_name: str = None, auto_refresh: bool = True):
        """
        Inicializa el almacén vectorial FAISS.
        
        Args:
            faiss_index_path (str): Ruta al archivo de índice FAISS (puede ser local o GCS)
            metadata_path (str): Ruta al archivo CSV con metadatos (puede ser local o GCS)
            threshold (float): Umbral mínimo de similitud para incluir resultados
            bucket_name (str): Nombre del bucket GCS (opcional, usa GCS_BUCKET_NAME por defecto)
            auto_refresh (bool): Si se debe refrescar automáticamente desde GCS
        """
        self.threshold = threshold
        self.bucket_name = bucket_name or GCS_BUCKET_NAME
        self.auto_refresh = auto_refresh
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        
        # Cargar datos
        self._load_data()
        
    def _load_data(self):
        """Carga los datos desde GCS o sistema de archivos local."""
        
        # Prioridad 1: Intentar usar el nuevo servicio GCS
        if USE_GCS and NEW_GCS_AVAILABLE and self.bucket_name:
            try:
                logger.info(f"Cargando datos desde GCS usando nuevo servicio: {self.bucket_name}")
                embeddings_data = load_embeddings_from_gcs(self.bucket_name)
                
                if embeddings_data.get('loaded_successfully', False):
                    self.index = embeddings_data['faiss_index']
                    self.metadata_df = embeddings_data['metadata']
                    self.config = embeddings_data.get('config', {})
                    
                    logger.info(f"Datos cargados exitosamente desde GCS (nuevo servicio)")
                    logger.info(f"Índice FAISS: {self.index.ntotal} vectores, dimensión {self.index.d}")
                    logger.info(f"Metadatos: {len(self.metadata_df)} registros")
                    return
                    
            except Exception as e:
                logger.error(f"Error al cargar con nuevo servicio GCS: {str(e)}")
                logger.info("Intentando con servicio legacy...")
        
        # Prioridad 2: Intentar usar el servicio GCS legacy
        if USE_GCS and GCS_AVAILABLE and self.bucket_name:
            try:
                logger.info(f"Cargando índice FAISS desde GCS (legacy): {self.bucket_name}")
                self.index, self.metadata_df = load_faiss_from_gcs(
                    self.bucket_name, 
                    os.path.basename(self.faiss_index_path),
                    os.path.basename(self.metadata_path)
                )
                
                logger.info(f"Datos cargados exitosamente desde GCS (legacy)")
                logger.info(f"Índice FAISS: {self.index.ntotal} vectores, dimensión {self.index.d}")
                logger.info(f"Metadatos: {len(self.metadata_df)} registros")
                return
                
            except Exception as e:
                logger.error(f"Error al cargar desde GCS (legacy): {str(e)}")
                logger.info("Intentando cargar desde archivos locales...")
        
        # Prioridad 3: Cargar desde sistema de archivos local
        try:
            logger.info(f"Cargando índice FAISS desde archivo local: {self.faiss_index_path}")
            
            if os.path.exists(self.faiss_index_path):
                self.index = faiss.read_index(self.faiss_index_path)
            else:
                raise FileNotFoundError(f"Archivo de índice FAISS no encontrado: {self.faiss_index_path}")
            
            # Cargar metadatos
            if os.path.exists(self.metadata_path):
                self.metadata_df = pd.read_csv(self.metadata_path)
            else:
                raise FileNotFoundError(f"Archivo de metadatos no encontrado: {self.metadata_path}")
            
            logger.info(f"Datos cargados exitosamente desde archivos locales")
            logger.info(f"Índice FAISS: {self.index.ntotal} vectores, dimensión {self.index.d}")
            logger.info(f"Metadatos: {len(self.metadata_df)} registros")
            
        except Exception as e:
            logger.error(f"Error al cargar desde archivos locales: {str(e)}")
            raise RuntimeError(f"No se pudieron cargar los datos desde ninguna fuente: {str(e)}")
    
    def refresh_from_gcs(self):
        """Refresca los datos desde Google Cloud Storage."""
        if USE_GCS and (NEW_GCS_AVAILABLE or GCS_AVAILABLE):
            logger.info("Refrescando datos desde Google Cloud Storage...")
            self._load_data()
        else:
            logger.warning("GCS no disponible, no se puede refrescar")
    
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
            
        # Auto-refresh si está habilitado
        if self.auto_refresh and USE_GCS:
            try:
                self.refresh_from_gcs()
            except Exception as e:
                logger.warning(f"Error al refrescar desde GCS: {str(e)}")
        
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

    def search_by_metadata(self, metadata_filter: Dict, limit: int = 5) -> List[Dict]:
        """
        Busca en el almacén por coincidencia exacta de metadatos.
        
        Args:
            metadata_filter (Dict): Diccionario con los filtros de metadatos (ejemplo: {'filename': 'documento.pdf'})
            limit (int): Número máximo de resultados a devolver
            
        Returns:
            List[Dict]: Lista de documentos que coinciden con los criterios
        """
        results = []
        
        # Verificar cada documento en los metadatos
        for idx, row in self.metadata_df.iterrows():
            # Verificar si todos los filtros coinciden
            match = True
            for key, value in metadata_filter.items():
                if key not in row or row[key] != value:
                    match = False
                    break
            
            # Si hay coincidencia, añadir a los resultados
            if match:
                metadata = row.to_dict()
                results.append(metadata)
                
                # Limitar el número de resultados
                if len(results) >= limit:
                    break
                    
        return results 
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas del almacén vectorial.
        
        Returns:
            Dict: Estadísticas del almacén
        """
        return {
            'total_vectors': self.index.ntotal,
            'vector_dimension': self.index.d,
            'metadata_records': len(self.metadata_df),
            'bucket_name': self.bucket_name,
            'using_gcs': USE_GCS and (NEW_GCS_AVAILABLE or GCS_AVAILABLE),
            'new_gcs_available': NEW_GCS_AVAILABLE,
            'legacy_gcs_available': GCS_AVAILABLE,
            'auto_refresh': self.auto_refresh
        }


# Función de conveniencia para crear un almacén vectorial
def create_vector_store(faiss_index_path: str = "data/embeddings/faiss_index.bin",
                       metadata_path: str = "data/embeddings/metadata.csv",
                       threshold: float = SIMILARITY_THRESHOLD,
                       bucket_name: str = None,
                       auto_refresh: bool = True) -> FAISSVectorStore:
    """
    Crea un almacén vectorial FAISS con configuración automática.
    
    Args:
        faiss_index_path (str): Ruta al índice FAISS
        metadata_path (str): Ruta a los metadatos
        threshold (float): Umbral de similitud
        bucket_name (str): Nombre del bucket GCS
        auto_refresh (bool): Auto-refresh desde GCS
        
    Returns:
        FAISSVectorStore: Almacén vectorial configurado
    """
    return FAISSVectorStore(
        faiss_index_path=faiss_index_path,
        metadata_path=metadata_path,
        threshold=threshold,
        bucket_name=bucket_name,
        auto_refresh=auto_refresh
    ) 