#!/usr/bin/env python3
"""
Script para configurar la base de datos PostgreSQL con pgvector.
Este script configura las tablas necesarias para el sistema RAG del chatbot UBA.
"""

import os
import sys
import logging
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from db.connection import test_connection, get_engine
from db.models import create_tables, get_table_info
from services.vector_db_service import VectorDBService

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Función principal para configurar la base de datos."""
    
    print("🚀 Configurando base de datos PostgreSQL para el chatbot UBA...")
    print("=" * 60)
    
    # 1. Probar conexión
    print("1. Probando conexión a la base de datos...")
    if not test_connection():
        print("❌ Error: No se pudo conectar a la base de datos")
        print("   Verifica que las variables de entorno estén configuradas:")
        print("   - DB_USER, DB_PASS, DB_NAME")
        print("   - CLOUD_SQL_CONNECTION_NAME (si usas Cloud SQL)")
        return False
    
    print("✅ Conexión establecida exitosamente")
    
    # 2. Crear tablas
    print("\n2. Creando tablas y extensiones...")
    try:
        engine = get_engine()
        create_tables(engine)
        print("✅ Tablas creadas exitosamente")
    except Exception as e:
        print(f"❌ Error al crear tablas: {str(e)}")
        return False
    
    # 3. Verificar tablas
    print("\n3. Verificando estructura de la base de datos...")
    try:
        table_info = get_table_info(engine)
        if table_info:
            print("📊 Información de la base de datos:")
            print(f"   - Extensión pgvector: {'✅' if table_info['vector_extension'] else '❌'}")
            print(f"   - Tabla embeddings: {'✅' if table_info['embeddings_table_exists'] else '❌'}")
            print(f"   - Tabla documents: {'✅' if table_info['documents_table_exists'] else '❌'}")
            print(f"   - Embeddings almacenados: {table_info['embeddings_count']}")
            print(f"   - Documentos almacenados: {table_info['documents_count']}")
        else:
            print("❌ No se pudo obtener información de las tablas")
            return False
    except Exception as e:
        print(f"❌ Error al verificar tablas: {str(e)}")
        return False
    
    # 4. Crear índices
    print("\n4. Creando índices para optimizar búsquedas...")
    try:
        vector_service = VectorDBService()
        if vector_service.create_index():
            print("✅ Índices creados exitosamente")
        else:
            print("⚠️ Warning: Algunos índices podrían no haberse creado")
    except Exception as e:
        print(f"❌ Error al crear índices: {str(e)}")
        return False
    
    # 5. Verificación final
    print("\n5. Verificación final del servicio...")
    try:
        vector_service = VectorDBService()
        if vector_service.test_connection():
            stats = vector_service.get_database_stats()
            print("✅ Servicio de base de datos vectorial funcionando correctamente")
            print(f"   - Total de embeddings: {stats.get('total_embeddings', 0)}")
            print(f"   - Documentos únicos: {stats.get('unique_documents', 0)}")
        else:
            print("❌ Error en la verificación final del servicio")
            return False
    except Exception as e:
        print(f"❌ Error en verificación final: {str(e)}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ¡Base de datos configurada exitosamente!")
    print("   El sistema RAG del chatbot UBA está listo para usar PostgreSQL con pgvector.")
    print("\n📝 Próximos pasos:")
    print("   1. Asegúrate de que tu archivo .env tenga las variables de Cloud SQL")
    print("   2. Ejecuta el chatbot para probar la nueva configuración")
    print("   3. Los embeddings se almacenarán automáticamente en PostgreSQL")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)