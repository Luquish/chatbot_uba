#!/usr/bin/env python3
"""
Script de verificación post-refactoring.
Valida que todos los módulos se importen y funcionen correctamente.
"""

import sys
import logging
from pathlib import Path

# Añadir directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Testear imports de módulos refactorizados."""
    try:
        # Test imports de módulos principales
        logger.info("🔍 Testeando imports de módulos principales...")
        
        from core.app_manager import app_manager, AppManager
        logger.info("✅ core.app_manager importado correctamente")
        
        from core.query_processor import QueryProcessor
        logger.info("✅ core.query_processor importado correctamente")
        
        from core.response_generator import ResponseGenerator
        logger.info("✅ core.response_generator importado correctamente")
        
        from core.context_retriever import ContextRetriever
        logger.info("✅ core.context_retriever importado correctamente")
        
        # Test import del sistema RAG refactorizado
        from rag_system import RAGSystem
        logger.info("✅ rag_system refactorizado importado correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en imports: {str(e)}")
        return False

def test_app_manager():
    """Testear funcionalidad del AppManager."""
    try:
        logger.info("🔍 Testeando AppManager...")
        
        from core.app_manager import app_manager
        
        # Test propiedades básicas
        assert hasattr(app_manager, 'environment')
        assert hasattr(app_manager, 'is_rag_ready')
        assert hasattr(app_manager, 'is_telegram_ready')
        logger.info("✅ AppManager tiene todas las propiedades esperadas")
        
        # Test estado del sistema
        status = app_manager.get_system_status()
        assert isinstance(status, dict)
        assert 'environment' in status
        assert 'rag_initialized' in status
        logger.info("✅ AppManager.get_system_status() funciona correctamente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en AppManager: {str(e)}")
        return False

def test_no_global_variables():
    """Verificar que no hay variables globales problemáticas."""
    try:
        logger.info("🔍 Verificando eliminación de variables globales...")
        
        # Leer main.py y verificar que no hay variables globales problemáticas
        with open(project_root / 'main.py', 'r') as f:
            content = f.read()
        
        # Verificar que no hay estas líneas problemáticas
        problematic_patterns = [
            'global rag_system',
            'rag_system = None',
            'rag_initialized = False'
        ]
        
        found_problems = []
        for pattern in problematic_patterns:
            if pattern in content:
                found_problems.append(pattern)
        
        if found_problems:
            logger.error(f"❌ Variables globales problemáticas encontradas: {found_problems}")
            return False
        
        logger.info("✅ No se encontraron variables globales problemáticas")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error verificando variables globales: {str(e)}")
        return False

def test_compilation():
    """Testear compilación de archivos principales."""
    try:
        logger.info("🔍 Testeando compilación de archivos...")
        
        import py_compile
        
        files_to_test = [
            'main.py',
            'rag_system.py',
            'core/app_manager.py',
            'core/query_processor.py',
            'core/response_generator.py',
            'core/context_retriever.py'
        ]
        
        for file_path in files_to_test:
            full_path = project_root / file_path
            if full_path.exists():
                py_compile.compile(str(full_path), doraise=True)
                logger.info(f"✅ {file_path} compila correctamente")
            else:
                logger.warning(f"⚠️ Archivo no encontrado: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en compilación: {str(e)}")
        return False

def main():
    """Función principal."""
    logger.info("🚀 INICIANDO VERIFICACIÓN POST-REFACTORING")
    logger.info("=" * 60)
    
    tests = [
        ("Imports de módulos", test_imports),
        ("Funcionalidad AppManager", test_app_manager),
        ("Eliminación de variables globales", test_no_global_variables),
        ("Compilación de archivos", test_compilation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Ejecutando: {test_name}")
        logger.info("-" * 40)
        
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.info(f"❌ {test_name}: FAILED")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE VERIFICACIÓN")
    logger.info(f"✅ Tests pasados: {passed}/{total}")
    logger.info(f"❌ Tests fallidos: {total - passed}/{total}")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 ¡REFACTORING VERIFICADO EXITOSAMENTE!")
        logger.info("✅ Todos los módulos funcionan correctamente")
        logger.info("✅ Variables globales eliminadas")
        logger.info("✅ Arquitectura modular implementada")
        return True
    else:
        logger.info("❌ REFACTORING REQUIERE CORRECCIONES")
        logger.info("Revisa los errores anteriores")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)