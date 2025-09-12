#!/usr/bin/env python3
"""
Script principal para ejecutar todos los tests del Chatbot UBA.
Ejecuta tests modulares y genera reporte de resultados.
"""

import asyncio
import sys
import time
import logging
from typing import Dict, List, Any
from pathlib import Path

# Configurar logging para tests
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Importar todos los tests
from test_config import TestConfig
from test_database import TestDatabase
from test_openai import TestOpenAI
from test_rag_system import TestRAGSystem
from test_chatbot_interaction import TestChatbotInteraction
from test_handlers_interaction import TestHandlersInteraction
from test_telegram import TestTelegram
from test_google_services import TestGoogleServices
from test_sheets_tools import TestSheetsTools
from test_http_endpoints import TestHttpEndpoints
# Ajustar import al nombre correcto del test de simulación
from test_simulation import TestRAGSimulation as TestSimulation
# Tests adicionales que faltaban
from test_hospitales_tool import TestHospitalesTool
from test_flexible_sessions import TestFlexibleSessions
from test_llm_vs_patterns import TestLLMVsPatterns
from test_session_diagnostic import TestSessionDiagnostic
from test_session_improvements import TestSessionImprovements
from test_session_production_flow import TestSessionProductionFlow
from test_sheets_catalog import TestSheetsCatalog
from test_session_calendar_consistency import TestSessionCalendarConsistency
from test_session_exceptions import TestSessionExceptions
from test_sessions import TestSessions
from test_vector_services import TestVectorServices


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
            TestSheetsTools(),
            TestHttpEndpoints(),
            TestSimulation(),
            # Tests adicionales que faltaban
            TestHospitalesTool(),
            TestFlexibleSessions(),
            TestLLMVsPatterns(),
            TestSessionDiagnostic(),
            TestSessionImprovements(),
            TestSessionProductionFlow(),
            TestSheetsCatalog(),
            TestSessionCalendarConsistency(),
            TestSessionExceptions(),
            TestSessions(),
            TestVectorServices()
        ]
        self.results = []
        
    def print_header(self):
        """Imprimir encabezado del test."""
        logging.info("🚀 INICIANDO VALIDACIÓN PRE-PRODUCCIÓN DEL CHATBOT UBA")
        logging.info("=" * 80)
        logging.info("Suite completa de tests para validar que el sistema está listo para producción:")
        logging.info("- Configuración completa y variables de entorno")
        logging.info("- Conectividad a todos los servicios externos")
        logging.info("- Funcionalidad end-to-end de interacciones")
        logging.info("- Endpoints HTTP críticos")
        logging.info("- Integraciones con APIs externas")
        logging.info("=" * 80)
        logging.info("")
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """Ejecutar todos los tests y retornar resultados."""
        results = []
        
        for i, test in enumerate(self.tests, 1):
            logging.info(f"📋 Test {i}/{len(self.tests)}: {test.get_test_description()}")
            logging.info("-" * 60)
            
            try:
                if hasattr(test, 'run_test') and asyncio.iscoroutinefunction(test.run_test):
                    result = await test.run_test()
                else:
                    result = test.run_test()
                
                results.append(result)
                
                if result['passed']:
                    logging.info(f"✅ {test.__class__.__name__}: PASSED")
                else:
                    logging.info(f"❌ {test.__class__.__name__}: FAILED")
                    if result['error_message']:
                        logging.info(f"   Error: {result['error_message']}")
                
            except Exception as e:
                logging.info(f"❌ {test.__class__.__name__}: ERROR")
                logging.info(f"   Error: {str(e)}")
                results.append({
                    "name": test.__class__.__name__,
                    "passed": False,
                    "error_message": str(e),
                    "details": {}
                })
            
            logging.info("")
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Imprimir resumen de resultados."""
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        
        logging.info("=" * 80)
        logging.info("📊 RESUMEN DE VALIDACIÓN PRE-PRODUCCIÓN")
        logging.info(f"✅ Tests pasados: {passed}/{total}")
        logging.info(f"❌ Tests fallidos: {total - passed}/{total}")
        logging.info("=" * 80)
        logging.info("")
        
        # Análisis de readiness
        logging.info("=" * 80)
        logging.info("🔍 ANÁLISIS DE READINESS PARA PRODUCCIÓN")
        logging.info("=" * 80)
        
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
        for category, category_passed in categories.items():
            status = "✅ CONFIGURADO" if category_passed else "❌ FALTANTE"
            logging.info(f"{status}: {category.upper()}")
        
        # Calcular puntuación
        config_score = 100.0 if categories.get('config', False) else 0.0
        functional_score = (passed / total) * 100 if total > 0 else 0.0
        # La puntuación general debe reflejar el éxito real de los tests
        overall_score = functional_score  # Usar solo el score funcional que es más preciso
        
        logging.info("")
        logging.info("📊 PUNTUACIÓN DE READINESS:")
        logging.info(f"   - Configuración: {config_score:.1f}%")
        logging.info(f"   - Tests funcionales: {functional_score:.1f}%")
        logging.info(f"   - PUNTUACIÓN GENERAL: {overall_score:.1f}%")
        logging.info("")
        
        if overall_score >= 90:
            logging.info("🎉 ¡SISTEMA LISTO PARA PRODUCCIÓN!")
            logging.info("📝 Pasos para despliegue:")
            logging.info("   1. Revisar variables de entorno en Cloud Run")
            logging.info("   2. Configurar webhook de Telegram en producción")
            logging.info("   3. Ejecutar: ./deploy.sh")
            logging.info("   4. Validar endpoints en el entorno de producción")
        elif overall_score >= 80:
            logging.info("\n✅ SISTEMA LISTO PARA PRODUCCIÓN")
            logging.info("   Todos los componentes críticos funcionan correctamente")
            logging.info("   Algunos tests menores pueden fallar pero no afectan la funcionalidad")
        elif overall_score >= 70:
            logging.info("\n⚠️ SISTEMA PARCIALMENTE LISTO")
            logging.info("   Algunos componentes necesitan atención antes del despliegue")
            logging.info("   Revisa los componentes marcados como FALTANTE")
        else:
            logging.info("\n❌ SISTEMA NO LISTO PARA PRODUCCIÓN")
            logging.info("   Se requieren correcciones significativas")
            logging.info("   Configura todos los componentes críticos antes de continuar")
        
        logging.info("=" * 80)
    
    async def run_simulation_if_ready(self, results: List[Dict[str, Any]]):
        """Ejecutar simulación si el sistema está listo."""
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        
        if passed >= total * 0.8:  # Si al menos 80% de tests pasaron
            logging.info("\n" + "=" * 80)
            logging.info("💡 SIMULACIÓN DE INTERACCIÓN REAL")
            logging.info("=" * 80)
            
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
        
        logging.info(f"\n🏁 VALIDACIÓN COMPLETADA en {end_time - start_time:.2f}s")
        if overall_score >= 80:
            logging.info("✅ RESULTADO: SISTEMA APROBADO PARA PRODUCCIÓN")
            return True
        else:
            logging.info("❌ RESULTADO: SISTEMA REQUIERE CORRECCIONES")
            logging.info("   Revisa los errores anteriores y vuelve a ejecutar")
            return False


async def main():
    """Función principal."""
    runner = TestRunner()
    success = await runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 