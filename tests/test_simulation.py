#!/usr/bin/env python3
"""
Test de simulación de interacción real.
Replica exactamente el flujo del webhook de WhatsApp.
"""

from base_test import AsyncBaseTest


class TestSimulation(AsyncBaseTest):
    """Test de simulación de interacción real."""
    
    def get_test_description(self) -> str:
        return "Simulación de interacción real del webhook de WhatsApp"
    
    def get_test_category(self) -> str:
        return "simulation"
    
    async def _run_test_logic(self) -> bool:
        """Simular interacción real."""
        print("🔄 Simulando mensaje entrante de WhatsApp...")
        
        # Simular un mensaje de WhatsApp como llega al webhook
        test_message = "¿Cómo presentar una denuncia en la facultad?"
        test_phone = "+541130897333"
        test_name = "Estudiante Test"
        
        print(f"📱 Mensaje recibido: '{test_message}'")
        print(f"📞 Número: {test_phone}")
        print(f"👤 Nombre: {test_name}")
        print()
        
        try:
            # Importar y inicializar los componentes necesarios
            from rag_system import RAGSystem
            from handlers.whatsapp_handler import WhatsAppHandler
            
            print("🔧 Inicializando componentes del sistema...")
            
            # Inicializar RAG System (como en main.py)
            rag_system = RAGSystem()
            print("✅ RAGSystem inicializado")
            
            # Inicializar WhatsApp Handler (como en main.py)
            whatsapp_handler = WhatsAppHandler(
                api_token=self.config.whatsapp.whatsapp_api_token,
                phone_number_id=self.config.whatsapp.whatsapp_phone_number_id,
                business_account_id=self.config.whatsapp.whatsapp_business_account_id
            )
            print("✅ WhatsApp Handler inicializado")
            print()
            
            # Simular el flujo exacto del webhook (como en main.py líneas 210-235)
            print("⚙️ Procesando mensaje...")
            
            # 1. Normalizar número (como en línea 211)
            normalized_from = whatsapp_handler.normalize_phone_number(test_phone)
            print(f"   📞 Número normalizado: {test_phone} -> {normalized_from}")
            
            # 2. Procesar con RAG (como en líneas 214-218)
            print(f"   🤖 Procesando consulta con RAG...")
            result = rag_system.process_query(
                test_message, 
                user_id=normalized_from,
                user_name=test_name
            )
            response_text = result["response"]
            
            # 3. Mostrar resultados
            print()
            print("📋 RESULTADO DE LA SIMULACIÓN:")
            print("=" * 50)
            print(f"👤 Usuario: {test_name} ({normalized_from})")
            print(f"💬 Mensaje: {test_message}")
            print()
            print("🤖 Respuesta del sistema:")
            print(f"   {response_text}")
            print()
            print(f"📊 Metadatos:")
            print(f"   - Tipo de consulta: {result.get('query_type', 'N/A')}")
            print(f"   - Chunks relevantes: {len(result.get('relevant_chunks', []))}")
            print(f"   - Fuentes utilizadas: {len(result.get('sources', []))}")
            if result.get('sources'):
                print(f"   - Fuentes: {', '.join(result['sources'][:3])}{'...' if len(result['sources']) > 3 else ''}")
            print("=" * 50)
            
            # 4. Simular el envío a WhatsApp (sin enviar realmente)
            print()
            print("📤 (En producción se enviaría via WhatsApp Business API)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en simulación: {str(e)}")
            print(f"   Tipo de error: {type(e).__name__}")
            return False 