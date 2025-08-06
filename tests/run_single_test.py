#!/usr/bin/env python3
"""
Script para ejecutar un test individual.
Uso: python run_single_test.py <nombre_del_test>
"""

import sys
import asyncio
from typing import Dict, Type
from base_test import BaseTest, AsyncBaseTest

# Importar todos los tests
from test_config import TestConfig
from test_database import TestDatabase
from test_openai import TestOpenAI
from test_rag_system import TestRAGSystem
from test_chatbot_interaction import TestChatbotInteraction
from test_handlers_interaction import TestHandlersInteraction
from test_whatsapp import TestWhatsApp
from test_google_services import TestGoogleServices
from test_http_endpoints import TestHttpEndpoints
from test_simulation import TestSimulation


# Mapeo de nombres de tests
TEST_MAP = {
    'config': TestConfig,
    'database': TestDatabase,
    'openai': TestOpenAI,
    'rag': TestRAGSystem,
    'interaction': TestChatbotInteraction,
    'handlers': TestHandlersInteraction,
    'whatsapp': TestWhatsApp,
    'google': TestGoogleServices,
    'http': TestHttpEndpoints,
    'simulation': TestSimulation
}


def print_available_tests():
    """Imprimir lista de tests disponibles."""
    print("Tests disponibles:")
    for name, test_class in TEST_MAP.items():
        print(f"  - {name}: {test_class().get_test_description()}")


async def run_single_test(test_name: str) -> bool:
    """Ejecutar un test específico."""
    if test_name not in TEST_MAP:
        print(f"❌ Test '{test_name}' no encontrado")
        print_available_tests()
        return False
    
    test_class = TEST_MAP[test_name]
    test_instance = test_class()
    
    print(f"🚀 Ejecutando test: {test_instance.get_test_description()}")
    print("=" * 60)
    
    try:
        if isinstance(test_instance, AsyncBaseTest):
            result = await test_instance.run_test()
        else:
            result = test_instance.run_test()
        
        if result['passed']:
            print(f"\n✅ {test_instance.__class__.__name__}: PASSED")
            return True
        else:
            print(f"\n❌ {test_instance.__class__.__name__}: FAILED")
            if result['error_message']:
                print(f"   Error: {result['error_message']}")
            return False
            
    except Exception as e:
        print(f"\n❌ {test_instance.__class__.__name__}: ERROR")
        print(f"   Error: {str(e)}")
        return False


async def main():
    """Función principal."""
    if len(sys.argv) != 2:
        print("Uso: python run_single_test.py <nombre_del_test>")
        print()
        print_available_tests()
        sys.exit(1)
    
    test_name = sys.argv[1].lower()
    success = await run_single_test(test_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 