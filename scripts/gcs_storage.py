#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para gestionar el acceso a archivos en Google Cloud Storage.
Este módulo permite leer archivos de embeddings directamente desde un bucket de GCS.
"""

import os
import logging
import tempfile
from pathlib import Path
from google.cloud import storage
import pandas as pd
import numpy as np
import faiss

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gcs_storage')

class GCSStorageManager:
    """
    Clase para gestionar el acceso a archivos en Google Cloud Storage.
    """
    
    def __init__(self, bucket_name: str = None):
        """
        Inicializa el gestor de almacenamiento de GCS.
        
        Args:
            bucket_name (str): Nombre del bucket de GCS. Si es None, se lee de la variable de entorno GCS_BUCKET_NAME.
        """
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("Se requiere el nombre del bucket de GCS (GCS_BUCKET_NAME)")
            
        # Inicializar cliente de GCS
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        logger.info(f"Gestor de almacenamiento de GCS inicializado para el bucket: {self.bucket_name}")
        
        # Directorio temporal para archivos descargados
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Directorio temporal creado: {self.temp_dir}")
    
    def list_files(self, prefix: str = "") -> list:
        """
        Lista los archivos en el bucket con un prefijo especificado.
        
        Args:
            prefix (str): Prefijo para filtrar archivos
            
        Returns:
            list: Lista de nombres de archivos
        """
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]
    
    def download_file(self, gcs_path: str, local_path: str = None) -> str:
        """
        Descarga un archivo de GCS a una ubicación local.
        
        Args:
            gcs_path (str): Ruta del archivo en GCS
            local_path (str): Ruta local donde guardar el archivo. Si es None, se crea una ruta temporal.
            
        Returns:
            str: Ruta local donde se descargó el archivo
        """
        if local_path is None:
            local_path = os.path.join(self.temp_dir, os.path.basename(gcs_path))
            
        blob = self.bucket.blob(gcs_path)
        
        if not blob.exists():
            logger.error(f"El archivo {gcs_path} no existe en el bucket {self.bucket_name}")
            raise FileNotFoundError(f"El archivo {gcs_path} no existe en el bucket {self.bucket_name}")
            
        logger.info(f"Descargando {gcs_path} a {local_path}")
        blob.download_to_filename(local_path)
        return local_path
    
    def read_file_to_string(self, gcs_path: str) -> str:
        """
        Lee un archivo de GCS como string.
        
        Args:
            gcs_path (str): Ruta del archivo en GCS
            
        Returns:
            str: Contenido del archivo como string
        """
        blob = self.bucket.blob(gcs_path)
        
        if not blob.exists():
            logger.error(f"El archivo {gcs_path} no existe en el bucket {self.bucket_name}")
            raise FileNotFoundError(f"El archivo {gcs_path} no existe en el bucket {self.bucket_name}")
            
        logger.info(f"Leyendo {gcs_path} como string")
        return blob.download_as_string().decode('utf-8')
    
    def read_binary_file(self, gcs_path: str) -> bytes:
        """
        Lee un archivo binario de GCS.
        
        Args:
            gcs_path (str): Ruta del archivo en GCS
            
        Returns:
            bytes: Contenido binario del archivo
        """
        blob = self.bucket.blob(gcs_path)
        
        if not blob.exists():
            logger.error(f"El archivo {gcs_path} no existe en el bucket {self.bucket_name}")
            raise FileNotFoundError(f"El archivo {gcs_path} no existe en el bucket {self.bucket_name}")
            
        logger.info(f"Leyendo {gcs_path} como binario")
        return blob.download_as_bytes()

def load_faiss_from_gcs(bucket_name: str = None, index_path: str = "faiss_index.bin", metadata_path: str = "metadata.csv"):
    """
    Carga un índice FAISS y sus metadatos desde GCS.
    
    Args:
        bucket_name (str): Nombre del bucket de GCS
        index_path (str): Ruta al archivo de índice FAISS en el bucket
        metadata_path (str): Ruta al archivo de metadatos en el bucket
        
    Returns:
        tuple: (índice FAISS, DataFrame de metadatos)
    """
    try:
        gcs_manager = GCSStorageManager(bucket_name)
        
        # Descargar el índice FAISS
        logger.info(f"Descargando índice FAISS desde gs://{gcs_manager.bucket_name}/{index_path}")
        index_blob = gcs_manager.bucket.blob(index_path)
        if not index_blob.exists():
            raise FileNotFoundError(f"No se encontró el índice FAISS en gs://{gcs_manager.bucket_name}/{index_path}")
        
        temp_index_file = os.path.join(gcs_manager.temp_dir, os.path.basename(index_path))
        index_blob.download_to_filename(temp_index_file)
        
        # Cargar el índice FAISS
        faiss_index = faiss.read_index(temp_index_file)
        logger.info(f"Índice FAISS cargado con {faiss_index.ntotal} vectores")
        
        # Descargar los metadatos
        logger.info(f"Descargando metadatos desde gs://{gcs_manager.bucket_name}/{metadata_path}")
        metadata_blob = gcs_manager.bucket.blob(metadata_path)
        if not metadata_blob.exists():
            raise FileNotFoundError(f"No se encontraron los metadatos en gs://{gcs_manager.bucket_name}/{metadata_path}")
        
        temp_metadata_file = os.path.join(gcs_manager.temp_dir, os.path.basename(metadata_path))
        metadata_blob.download_to_filename(temp_metadata_file)
        
        # Cargar los metadatos
        metadata_df = pd.read_csv(temp_metadata_file)
        logger.info(f"Metadatos cargados con {len(metadata_df)} registros")
        
        return faiss_index, metadata_df
        
    except Exception as e:
        logger.error(f"Error al cargar índice FAISS desde GCS: {str(e)}")
        raise

if __name__ == "__main__":
    """
    Código de prueba para verificar la funcionalidad del módulo.
    """
    try:
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        print(f"Probando acceso al bucket: {bucket_name}")
        
        gcs_manager = GCSStorageManager(bucket_name)
        files = gcs_manager.list_files()
        print(f"Archivos en el bucket: {files}")
        
        # Probar carga de índice FAISS
        if 'faiss_index.bin' in files and 'metadata.csv' in files:
            print("Probando carga de índice FAISS...")
            index, metadata = load_faiss_from_gcs(bucket_name)
            print(f"Índice FAISS cargado con {index.ntotal} vectores")
            print(f"Metadatos cargados con {len(metadata)} registros")
            print("Pruebas completadas exitosamente")
    
    except Exception as e:
        print(f"Error en las pruebas: {str(e)}") 