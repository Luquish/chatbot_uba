#!/usr/bin/env python3
"""
Test de integración de Telegram Bot API.
Valida el handler y envío de mensajes.
"""

import asyncio
from base_test import BaseTest


class TestTelegram(BaseTest):
    """Test de integración de Telegram."""
    
    def get_test_description(self) -> str:
        return "Test de integración completa de Telegram Bot API"
    
    def get_test_category(self) -> str:
        return "telegram"
    
    def _run_test_logic(self) -> bool:
        """Validar integración de Telegram."""
        print("🤖 Probando integración completa de Telegram...")
        
        try:
            from handlers.telegram_handler import TelegramHandler
            
            # Verificar configuración
            if not self.config.telegram.telegram_bot_token:
                self.log_error("TELEGRAM_BOT_TOKEN no configurado")
                return False
            
            # Inicializar Telegram handler
            telegram_handler = TelegramHandler(
                bot_token=self.config.telegram.telegram_bot_token
            )
            self.log_success("Telegram handler inicializado")
            
            # Test de obtención de información del bot (síncrono)
            bot_info = asyncio.run(telegram_handler.get_me())
            if bot_info.get("ok"):
                bot_data = bot_info.get("result", {})
                print(f"   Bot ID: {bot_data.get('id')}")
                print(f"   Username: @{bot_data.get('username')}")
                print(f"   Nombre: {bot_data.get('first_name')}")
                self.log_success("Información del bot obtenida correctamente")
            else:
                self.log_error(f"Error al obtener info del bot: {bot_info}")
                return False
            
            # Test de webhook info
            webhook_info = asyncio.run(telegram_handler.get_webhook_info())
            if webhook_info.get("ok"):
                webhook_data = webhook_info.get("result", {})
                print(f"   Webhook URL: {webhook_data.get('url', 'No configurado')}")
                print(f"   Pending updates: {webhook_data.get('pending_update_count', 0)}")
                self.log_success("Información del webhook obtenida")
            else:
                self.log_warning("No se pudo obtener información del webhook")
            
            # Test con usuario específico (si está configurado)
            if self.config.telegram.telegram_admin_user_id:
                test_message = "🧪 Este es un mensaje de prueba del sistema RAG UBA\n\n✅ Test de integración completado exitosamente."
                test_result = asyncio.run(telegram_handler.send_message(
                    self.config.telegram.telegram_admin_user_id,
                    test_message
                ))
                
                if test_result.get('status') == 'success':
                    self.log_success("Mensaje de prueba enviado exitosamente")
                    print(f"   Message ID: {test_result.get('message_id')}")
                else:
                    self.log_error(f"Error al enviar mensaje: {test_result}")
                    return False
            else:
                self.log_warning("No hay TELEGRAM_ADMIN_USER_ID configurado para pruebas")
            
            return True
            
        except Exception as e:
            self.log_error(f"Error en test de Telegram: {str(e)}")
            return False


if __name__ == "__main__":
    test = TestTelegram()
    success = test.run_test()
    exit(0 if success else 1)
