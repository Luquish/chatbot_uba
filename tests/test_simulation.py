#!/usr/bin/env python3
"""
Test de simulación de interacción real con RAG.
Replica exactamente el flujo del webhook de Telegram + RAG.
"""

from base_test import AsyncBaseTest


class TestRAGSimulation(AsyncBaseTest):
    """Test de simulación de interacción real con RAG y Telegram."""
    
    def get_test_description(self) -> str:
        return "Simulación completa de RAG + Telegram - Test de flujo de consultas"
    
    def get_test_category(self) -> str:
        return "rag_telegram"
    
    async def _run_test_logic(self) -> bool:
        """Simular interacción real con RAG."""
        print("🔄 Simulando mensaje entrante de Telegram con RAG...")
        
        # Simular varios mensajes de prueba para validar RAG
        test_cases = [
            {
                "message": "¿Cómo presentar una denuncia en la facultad?",
                "user_id": "123456789",
                "user_name": "Estudiante Test",
                "chat_id": "123456789",
                "expected_topics": ["denuncia", "procedimiento", "administrativo"]
            },
            {
                "message": "¿Qué actividades hay esta semana?",
                "user_id": "987654321", 
                "user_name": "María González",
                "chat_id": "987654321",
                "expected_topics": ["eventos", "actividades", "semana"]
            },
            {
                "message": "Mostrame las próximas actividades",
                "user_id": "456789123",
                "user_name": "Juan Pérez", 
                "chat_id": "456789123",
                "expected_topics": ["eventos", "actividades", "próximos"]
            }
        ]
        
        try:
            # Importar y inicializar los componentes necesarios
            from rag_system import RAGSystem
            from handlers.telegram_handler import TelegramHandler
            
            print("🔧 Inicializando componentes del sistema...")
            
            # Inicializar RAG System (como en main.py)
            rag_system = RAGSystem()
            print("✅ RAGSystem inicializado")
            
            # Inicializar Telegram Handler (como en main.py)
            if self.config.telegram.telegram_bot_token:
                telegram_handler = TelegramHandler(
                    bot_token=self.config.telegram.telegram_bot_token
                )
                print("✅ Telegram Handler inicializado")
            else:
                print("⚠️ TELEGRAM_BOT_TOKEN no configurado - usando simulación")
                telegram_handler = None
            print()
            
            successful_cases = 0
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"📋 CASO DE PRUEBA {i}/{len(test_cases)}")
                print("=" * 50)
                
                # Simular el flujo exacto del webhook
                print("⚙️ Procesando mensaje...")
                print(f"📱 Mensaje: '{test_case['message']}'")
                print(f"👤 Usuario: {test_case['user_name']} (ID: {test_case['user_id']})")
                print(f"💬 Chat ID: {test_case['chat_id']}")
                print()
                
                # Procesar con RAG (como en main.py líneas 154-160)
                print(f"🤖 Procesando consulta con RAG...")
                result = rag_system.process_query(
                    test_case['message'], 
                    user_id=test_case['user_id'],
                    user_name=test_case['user_name']
                )
                response_text = result["response"]
                
                # Analizar resultados
                print()
                print("🤖 Respuesta del sistema:")
                print(f"   {response_text}")
                print()
                print(f"📊 Metadatos RAG:")
                print(f"   - Tipo de consulta: {result.get('query_type', 'N/A')}")
                print(f"   - Chunks relevantes: {len(result.get('relevant_chunks', []))}")
                print(f"   - Fuentes utilizadas: {len(result.get('sources', []))}")
                if result.get('sources'):
                    print(f"   - Fuentes: {', '.join(result['sources'][:3])}{'...' if len(result['sources']) > 3 else ''}")
                
                # Validar que la respuesta es relevante
                response_lower = response_text.lower()
                found_topics = [topic for topic in test_case['expected_topics'] 
                              if topic in response_lower]
                
                if found_topics:
                    print(f"   ✅ Temas encontrados: {', '.join(found_topics)}")
                    successful_cases += 1
                    case_success = True
                else:
                    print(f"   ⚠️ Temas esperados no encontrados: {', '.join(test_case['expected_topics'])}")
                    case_success = False
                
                # Simular envío a Telegram (sin enviar realmente)
                print()
                if telegram_handler:
                    print("📤 (En producción se enviaría via Telegram Bot API)")
                else:
                    print("📤 (Simulación - TELEGRAM_BOT_TOKEN no configurado)")
                
                print(f"🎯 Resultado del caso: {'✅ Exitoso' if case_success else '❌ Falló'}")
                print("=" * 50)
                print()
            
            # Resumen final
            success_rate = (successful_cases / len(test_cases)) * 100
            print(f"📊 RESUMEN DE PRUEBAS RAG + TELEGRAM:")
            print(f"   Casos exitosos: {successful_cases}/{len(test_cases)} ({success_rate:.1f}%)")
            print(f"   Sistema RAG: {'✅ Funcionando' if successful_cases > 0 else '❌ Problemas'}")
            print(f"   Telegram Handler: {'✅ Configurado' if telegram_handler else '⚠️ No configurado'}")
            
            # Consideramos exitoso si al menos 2/3 de los casos funcionan
            return successful_cases >= (len(test_cases) * 0.67)
            
        except Exception as e:
            print(f"❌ Error en simulación RAG: {str(e)}")
            print(f"   Tipo de error: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    test = TestRAGSimulation()
    import asyncio
    success = asyncio.run(test.run_test())
    exit(0 if success else 1) 