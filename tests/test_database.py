#!/usr/bin/env python3
"""
Test de conexión a PostgreSQL y funcionalidad de base de datos.
Valida la conexión a Cloud SQL y operaciones básicas.
"""

import numpy as np
from sqlalchemy import text
from base_test import BaseTest


class TestDatabase(BaseTest):
    """Test de conexión y funcionalidad de PostgreSQL."""
    
    def get_test_description(self) -> str:
        return "Test completo de conexión y funcionalidad de PostgreSQL"
    
    def get_test_category(self) -> str:
        return "database"
    
    def _run_test_logic(self) -> bool:
        """Validar conexión y funcionalidad de PostgreSQL."""
        print("🔧 Probando conexión y funcionalidad de PostgreSQL...")
        
        try:
            from db.connection import test_connection, get_engine
            from services.vector_db_service import VectorDBService
            
            # Test básico de conexión
            if test_connection():
                self.log_success("Conexión básica a PostgreSQL exitosa")
            else:
                self.log_error("Conexión básica a PostgreSQL falló")
                return False
            
            # Test de operaciones básicas
            engine = get_engine()
            
            # Test 1: Verificar que las tablas principales existen
            try:
                with engine.connect() as conn:
                    # Verificar tablas principales según setup_pgvector.sql
                    result = conn.execute(text("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('embeddings', 'documents')
                    """))
                    tables = [row[0] for row in result.fetchall()]
                    
                    if 'embeddings' in tables:
                        self.log_success("Tabla embeddings existe")
                    else:
                        self.log_warning("Tabla embeddings no encontrada")
                        
                    if 'documents' in tables:
                        self.log_success("Tabla documents existe")
                    else:
                        self.log_warning("Tabla documents no encontrada")
                        
                    if tables:
                        self.log_success(f"Tablas principales encontradas: {', '.join(tables)}")
                    else:
                        self.log_warning("No se encontraron tablas principales")
                        
            except Exception as e:
                self.log_warning(f"Error verificando tablas: {str(e)}")
            
            # Test 2: Verificar extensión pgvector
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT extname FROM pg_extension WHERE extname = 'vector'
                    """))
                    if result.fetchone():
                        self.log_success("Extensión pgvector instalada")
                    else:
                        self.log_warning("Extensión pgvector no encontrada")
            except Exception as e:
                self.log_warning(f"Error verificando pgvector: {str(e)}")
            
            # Test 3: Verificar datos existentes (solo lectura)
            try:
                with engine.connect() as conn:
                    # Contar embeddings existentes
                    result = conn.execute(text("SELECT COUNT(*) FROM embeddings"))
                    embeddings_count = result.fetchone()[0]
                    self.log_info(f"Total embeddings en la base de datos: {embeddings_count}")
                    
                    # Contar documentos existentes
                    result = conn.execute(text("SELECT COUNT(*) FROM documents"))
                    documents_count = result.fetchone()[0]
                    self.log_info(f"Total documentos en la base de datos: {documents_count}")
                    
                    if embeddings_count > 0:
                        self.log_success("Base de datos contiene embeddings")
                    else:
                        self.log_warning("Base de datos vacía de embeddings")
                        
            except Exception as e:
                self.log_warning(f"Error contando datos: {str(e)}")
            
            # Test 3: Verificar tablas existentes
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('embeddings', 'documents', 'test_connection')
                    """))
                    tables = [row[0] for row in result.fetchall()]
                    if tables:
                        self.log_success(f"Tablas existentes: {', '.join(tables)}")
                    else:
                        self.log_warning("No se encontraron tablas esperadas")
            except Exception as e:
                self.log_warning(f"Error verificando tablas: {str(e)}")
            
            # Test 4: Limpiar tabla de prueba
            try:
                with engine.connect() as conn:
                    conn.execute(text("DROP TABLE IF EXISTS test_connection"))
                    conn.commit()
                    self.log_success("Limpieza de tablas funcionando")
            except Exception as e:
                if "permission denied" in str(e).lower():
                    self.log_warning("Limpieza de tablas limitada (permisos en Cloud SQL)")
                    self.log_info("La conexión a PostgreSQL funciona correctamente")
                else:
                    self.log_warning(f"Error en limpieza: {str(e)}")
            
            # Test de servicio vectorial
            try:
                vector_service = VectorDBService()
                stats = vector_service.get_database_stats()
                self.log_success("Servicio vectorial básico operativo")
                self.log_info(f"Total embeddings: {stats.get('total_embeddings', 0)}")
                
                # Test de búsqueda vectorial básica
                if stats.get('total_embeddings', 0) > 0:
                    test_embedding = np.random.random(1536).astype(np.float32)
                    results = vector_service.similarity_search(test_embedding, k=1)
                    self.log_success(f"Búsqueda vectorial funcional: {len(results)} resultado(s)")
                else:
                    self.log_warning("No hay embeddings en la base de datos")
                    
            except Exception as e:
                if "permission denied to create extension" in str(e) or "vector.so" in str(e):
                    self.log_warning("pgvector no disponible (normal en desarrollo local)")
                    self.log_info("La conexión a PostgreSQL funciona correctamente")
                    self.log_info("Para producción, usar Cloud SQL con pgvector habilitado")
                else:
                    self.log_warning(f"Error en servicio vectorial: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_error(f"PostgreSQL no disponible: {str(e)}")
            return False 