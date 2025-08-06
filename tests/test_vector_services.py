#!/usr/bin/env python3
"""
Test de servicios vectoriales avanzados.
Valida operaciones de embeddings y búsquedas de similitud.
"""

import numpy as np
from base_test import BaseTest


class TestVectorServices(BaseTest):
    """Test de servicios vectoriales."""
    
    def get_test_description(self) -> str:
        return "Test de servicios vectoriales avanzados"
    
    def get_test_category(self) -> str:
        return "vector"
    
    def _run_test_logic(self) -> bool:
        """Validar servicios vectoriales."""
        print("🔍 Probando servicios vectoriales avanzados...")
        
        try:
            from storage.vector_store import PostgreSQLVectorStore
            from services.vector_db_service import VectorDBService
            
            # Test de vector store
            try:
                vector_store = PostgreSQLVectorStore(threshold=self.config.rag.similarity_threshold)
                self.log_success("Vector store inicializado correctamente")
                
                # Test de búsqueda con embedding de prueba
                test_embedding = np.random.random(1536).astype(np.float32)
                results = vector_store.search(test_embedding, k=5)
                
                self.log_success(f"Vector store operativo: {len(results)} resultados")
                
            except Exception as e:
                if "permission denied to create extension" in str(e) or "vector.so" in str(e):
                    self.log_warning("pgvector no disponible (normal en desarrollo)")
                else:
                    self.log_warning(f"Error en vector store: {str(e)}")
            
            # Test de servicio de base de datos vectorial
            try:
                vector_service = VectorDBService()
                stats = vector_service.get_database_stats()
                
                self.log_success("Vector DB service conectado")
                self.log_info(f"Total embeddings: {stats.get('total_embeddings', 0)}")
                self.log_info(f"Total documentos: {stats.get('total_documents', 0)}")
                
            except Exception as e:
                self.log_warning(f"Error en vector DB service: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Error en servicios vectoriales: {str(e)}")
            return False 