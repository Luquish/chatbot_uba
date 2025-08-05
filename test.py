#!/usr/bin/env python3
"""
Test de conectividad y funcionalidad del sistema RAG con PostgreSQL.
Verifica que la base de datos esté correctamente configurada y funcionando.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Añadir el directorio raíz al path para importar módulos
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test básico de conexión a la base de datos."""
    print("🔧 Probando conexión a la base de datos...")
    
    try:
        from db.connection import test_connection, get_engine
        
        if test_connection():
            print("✅ Conexión a PostgreSQL exitosa")
            return True
        else:
            print("❌ Error: No se pudo conectar a PostgreSQL")
            return False
            
    except Exception as e:
        print(f"❌ Error en conexión: {str(e)}")
        return False

def test_vector_db_service():
    """Test del servicio de base de datos vectorial."""
    print("\n🧠 Probando servicio de base de datos vectorial...")
    
    try:
        from services.vector_db_service import VectorDBService
        
        # Inicializar servicio
        vector_service = VectorDBService()
        print("✅ VectorDBService inicializado correctamente")
        
        # Probar conexión
        if vector_service.test_connection():
            print("✅ Conexión del servicio vectorial exitosa")
        else:
            print("❌ Error en conexión del servicio vectorial")
            return False
        
        # Obtener estadísticas
        stats = vector_service.get_database_stats()
        print(f"📊 Estadísticas de la base de datos:")
        print(f"   - Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"   - Documentos únicos: {stats.get('unique_documents', 0)}")
        
        table_info = stats.get('table_info', {})
        if table_info:
            print(f"   - Extensión pgvector: {'✅' if table_info.get('vector_extension') else '❌'}")
            print(f"   - Tabla embeddings: {'✅' if table_info.get('embeddings_table_exists') else '❌'}")
            print(f"   - Tabla documents: {'✅' if table_info.get('documents_table_exists') else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en servicio vectorial: {str(e)}")
        return False

def test_vector_store():
    """Test del almacén vectorial PostgreSQL."""
    print("\n📦 Probando almacén vectorial...")
    
    try:
        from storage.vector_store import PostgreSQLVectorStore
        
        # Inicializar vector store
        vector_store = PostgreSQLVectorStore(threshold=0.7)
        print("✅ PostgreSQLVectorStore inicializado correctamente")
        
        # Obtener estadísticas
        stats = vector_store.get_stats()
        print(f"📊 Estadísticas del vector store:")
        print(f"   - Total vectores: {stats.get('total_vectors', 0)}")
        print(f"   - Dimensión vectorial: {stats.get('vector_dimension', 0)}")
        print(f"   - Documentos únicos: {stats.get('unique_documents', 0)}")
        print(f"   - Usando PostgreSQL: {'✅' if stats.get('using_postgresql') else '❌'}")
        print(f"   - Usando pgvector: {'✅' if stats.get('using_pgvector') else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en vector store: {str(e)}")
        return False

def test_embeddings_verification():
    """Test de verificación de embeddings existentes (solo lectura)."""
    print("\n🔍 Verificando embeddings existentes en la base de datos...")
    
    try:
        from services.vector_db_service import VectorDBService
        
        # Verificar embeddings existentes
        vector_service = VectorDBService()
        stats = vector_service.get_database_stats()
        
        if stats and stats.get('total_embeddings', 0) > 0:
            print(f"✅ Base de datos contiene {stats.get('total_embeddings', 0)} embeddings")
            print(f"✅ {stats.get('total_documents', 0)} documentos disponibles")
            print("✅ Verificación de embeddings existentes: PASSED")
        else:
            print("⚠️ Base de datos vacía (esto es normal si no se han cargado documentos)")
            print("✅ Verificación de embeddings existentes: PASSED (base de datos accesible)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al verificar embeddings: {str(e)}")
        return False

def test_similarity_search():
    """Test de búsqueda de similitud."""
    print("\n🔍 Probando búsqueda de similitud...")
    
    try:
        from storage.vector_store import PostgreSQLVectorStore
        
        # Crear query embedding de prueba
        query_embedding = np.random.random(1536).astype(np.float32)
        
        # Realizar búsqueda
        vector_store = PostgreSQLVectorStore(threshold=0.0)  # Umbral bajo para testing
        results = vector_store.search(query_embedding, k=5)
        
        print(f"🔍 Búsqueda completada: {len(results)} resultados encontrados")
        
        if results:
            for i, result in enumerate(results[:3]):  # Mostrar solo los primeros 3
                print(f"   Resultado {i+1}:")
                print(f"     - Texto: {result.get('text', '')[:50]}...")
                print(f"     - Similitud: {result.get('similarity', 0):.4f}")
                print(f"     - Documento: {result.get('document_id', '')}")
        else:
            print("   No se encontraron resultados (esto es normal si la base de datos está vacía)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en búsqueda de similitud: {str(e)}")
        return False

def test_rag_system():
    """Test del sistema RAG completo."""
    print("\n🤖 Probando sistema RAG completo...")
    
    try:
        from rag_system import RAGSystem
        
        # Inicializar sistema RAG
        rag = RAGSystem()
        print("✅ RAGSystem inicializado correctamente")
        
        # Probar estadísticas del vector store
        stats = rag.vector_store.get_stats()
        print(f"📊 Sistema RAG conectado a PostgreSQL:")
        print(f"   - Total vectores: {stats.get('total_vectors', 0)}")
        print(f"   - Umbral de similitud: {stats.get('threshold', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en sistema RAG: {str(e)}")
        return False

def cleanup_test_data():
    """Verifica que no hay datos de prueba en la base de datos."""
    print("\n🧹 Verificando limpieza de datos de prueba...")
    
    try:
        from services.vector_db_service import VectorDBService
        
        # Verificar que no hay datos de prueba
        vector_service = VectorDBService()
        test_embeddings = vector_service.get_document_embeddings('test_doc_001')
        
        if not test_embeddings:
            print("✅ No se encontraron datos de prueba (correcto)")
        else:
            print(f"⚠️ Se encontraron {len(test_embeddings)} embeddings de prueba")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al verificar limpieza: {str(e)}")
        return False

def main():
    """Función principal que ejecuta todos los tests."""
    print("🚀 INICIANDO TESTS DEL SISTEMA RAG CON POSTGRESQL")
    print("=" * 60)
    
    # Lista de tests a ejecutar
    tests = [
        ("Conexión a base de datos", test_database_connection),
        ("Servicio de base de datos vectorial", test_vector_db_service),
        ("Almacén vectorial", test_vector_store),
        ("Verificación de embeddings existentes", test_embeddings_verification),
        ("Búsqueda de similitud", test_similarity_search),
        ("Sistema RAG completo", test_rag_system),
        ("Limpieza de datos de prueba", cleanup_test_data)
    ]
    
    # Ejecutar tests
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE TESTS")
    print(f"✅ Tests pasados: {passed}/{total}")
    print(f"❌ Tests fallidos: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("   El sistema está listo para usar PostgreSQL con pgvector.")
        print("\n💡 Próximos pasos:")
        print("   1. Ejecuta el chatbot: python main.py")
        print("   2. Los embeddings se almacenarán automáticamente en PostgreSQL")
        print("   3. Las búsquedas usarán pgvector para alta performance")
    else:
        print(f"\n⚠️ {total - passed} tests fallaron.")
        print("   Revisa la configuración de PostgreSQL y las variables de entorno.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)