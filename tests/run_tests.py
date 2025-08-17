#!/usr/bin/env python3
"""
Script principal para ejecutar todos los tests del Chatbot UBA.
Ejecuta tests modulares y genera reporte de resultados.
"""

import asyncio
import sys
import time
from typing import Dict, List, Any
from pathlib import Path

# Importar todos los tests
from test_config import TestConfig
from test_database import TestDatabase
from test_openai import TestOpenAI
from test_rag_system import TestRAGSystem
from test_chatbot_interaction import TestChatbotInteraction
from test_handlers_interaction import TestHandlersInteraction
from test_telegram import TestTelegram
from test_google_services import TestGoogleServices
from test_http_endpoints import TestHttpEndpoints
from test_simulation import TestSimulation


class TestRunner:
    """Ejecutor de tests con reporte de resultados."""
    
    def __init__(self):
        """Inicializar ejecutor de tests."""
        self.tests = [
            TestConfig(),
            TestDatabase(),
            TestOpenAI(),
            TestRAGSystem(),
            TestChatbotInteraction(),
            TestHandlersInteraction(),
            TestTelegram(),
            TestGoogleServices(),
            TestHttpEndpoints(),
            TestSimulation()
        ]
        self.results = []
        
    def print_header(self):
        """Imprimir encabezado del test."""
        print("🚀 INICIANDO VALIDACIÓN PRE-PRODUCCIÓN DEL CHATBOT UBA")
        print("=" * 80)
        print("Suite completa de tests para validar que el sistema está listo para producción:")
        print("- Configuración completa y variables de entorno")
        print("- Conectividad a todos los servicios externos")
        print("- Funcionalidad end-to-end de interacciones")
        print("- Endpoints HTTP críticos")
        print("- Integraciones con APIs externas")
        print("=" * 80)
        print()
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Ejecutar todos los tests y retornar resultados."""
        results = []
        
        for i, test in enumerate(self.tests, 1):
            print(f"📋 Test {i}/{len(self.tests)}: {test.get_test_description()}")
            print("-" * 60)
            
            try:
                if hasattr(test, 'run_test') and asyncio.iscoroutinefunction(test.run_test):
                    result = await test.run_test()
                else:
                    result = test.run_test()
                
                results.append(result)
                
                if result['passed']:
                    print(f"✅ {test.__class__.__name__}: PASSED")
                else:
                    print(f"❌ {test.__class__.__name__}: FAILED")
                    if result['error_message']:
                        print(f"   Error: {result['error_message']}")
                
            except Exception as e:
                print(f"❌ {test.__class__.__name__}: ERROR")
                print(f"   Error: {str(e)}")
                results.append({
                    "name": test.__class__.__name__,
                    "passed": False,
                    "error_message": str(e),
                    "details": {}
                })
            
            print()
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Imprimir resumen de resultados."""
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        
        print("=" * 80)
        print("📊 RESUMEN DE VALIDACIÓN PRE-PRODUCCIÓN")
        print(f"✅ Tests pasados: {passed}/{total}")
        print(f"❌ Tests fallidos: {total - passed}/{total}")
        print("=" * 80)
        print()
        
        # Análisis de readiness
        print("=" * 80)
        print("🔍 ANÁLISIS DE READINESS PARA PRODUCCIÓN")
        print("=" * 80)
        
        # Categorizar resultados
        categories = {}
        for result in results:
            test_name = result['name']
            if 'Config' in test_name:
                categories['config'] = result['passed']
            elif 'Database' in test_name or 'Vector' in test_name:
                categories['database'] = result['passed']
            elif 'OpenAI' in test_name or 'RAG' in test_name:
                categories['ai'] = result['passed']
            elif 'Telegram' in test_name or 'Interaction' in test_name:
                categories['telegram'] = result['passed']
            elif 'Google' in test_name:
                categories['google'] = result['passed']
            elif 'Http' in test_name:
                categories['http'] = result['passed']
            elif 'Simulation' in test_name:
                categories['simulation'] = result['passed']
        
        # Mostrar estado por categoría
        for category, passed in categories.items():
            status = "✅ CONFIGURADO" if passed else "❌ FALTANTE"
            print(f"{status}: {category.upper()}")
        
        # Calcular puntuación
        config_score = 100.0 if categories.get('config', False) else 0.0
        functional_score = (passed / total) * 100 if total > 0 else 0.0
        overall_score = (config_score + functional_score) / 2
        
        print()
        print("📊 PUNTUACIÓN DE READINESS:")
        print(f"   - Configuración: {config_score:.1f}%")
        print(f"   - Tests funcionales: {functional_score:.1f}%")
        print(f"   - PUNTUACIÓN GENERAL: {overall_score:.1f}%")
        print()
        
        if overall_score >= 90:
            print("🎉 ¡SISTEMA LISTO PARA PRODUCCIÓN!")
            print("📝 Pasos para despliegue:")
            print("   1. Revisar variables de entorno en Cloud Run")
            print("   2. Configurar webhook de Telegram en producción")
            print("   3. Ejecutar: ./deploy.sh")
            print("   4. Validar endpoints en el entorno de producción")
        elif overall_score >= 70:
            print("\n⚠️ SISTEMA PARCIALMENTE LISTO")
            print("   Algunos componentes necesitan atención antes del despliegue")
            print("   Revisa los componentes marcados como FALTANTE")
        else:
            print("\n❌ SISTEMA NO LISTO PARA PRODUCCIÓN")
            print("   Se requieren correcciones significativas")
            print("   Configura todos los componentes críticos antes de continuar")
        
        print("=" * 80)
    
    async def run_simulation_if_ready(self, results: List[Dict[str, Any]]):
        """Ejecutar simulación si el sistema está listo."""
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        
        if passed >= total * 0.8:  # Si al menos 80% de tests pasaron
            print("\n" + "=" * 80)
            print("💡 SIMULACIÓN DE INTERACCIÓN REAL")
            print("=" * 80)
            
            # Ejecutar simulación
            simulation = TestSimulation()
            await simulation.run_test()
    
    async def run(self) -> bool:
        """Ejecutar suite completa de tests."""
        self.print_header()
        
        start_time = time.time()
        results = await self.run_all_tests()
        end_time = time.time()
        
        self.print_summary(results)
        await self.run_simulation_if_ready(results)
        
        # Determinar si el sistema está listo
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        overall_score = (passed / total) * 100 if total > 0 else 0
        
        print(f"\n🏁 VALIDACIÓN COMPLETADA en {end_time - start_time:.2f}s")
        if overall_score >= 90:
            print("✅ RESULTADO: SISTEMA APROBADO PARA PRODUCCIÓN")
            return True
        else:
            print("❌ RESULTADO: SISTEMA REQUIERE CORRECCIONES")
            print("   Revisa los errores anteriores y vuelve a ejecutar")
            return False


async def main():
    """Función principal."""
    runner = TestRunner()
    success = await runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 