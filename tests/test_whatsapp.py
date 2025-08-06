#!/usr/bin/env python3
"""
Test de integración de WhatsApp Business API.
Valida el handler y normalización de números.
"""

from base_test import BaseTest


class TestWhatsApp(BaseTest):
    """Test de integración de WhatsApp."""
    
    def get_test_description(self) -> str:
        return "Test de integración completa de WhatsApp"
    
    def get_test_category(self) -> str:
        return "whatsapp"
    
    def _run_test_logic(self) -> bool:
        """Validar integración de WhatsApp."""
        print("📱 Probando integración completa de WhatsApp...")
        
        try:
            from handlers.whatsapp_handler import WhatsAppHandler
            
            # Inicializar WhatsApp handler
            whatsapp_handler = WhatsAppHandler(
                api_token=self.config.whatsapp.whatsapp_api_token,
                phone_number_id=self.config.whatsapp.whatsapp_phone_number_id,
                business_account_id=self.config.whatsapp.whatsapp_business_account_id
            )
            self.log_success("WhatsApp handler inicializado")
            
            # Test de normalización de números
            test_numbers = [
                "+54911234567890",
                "549112345678",
                "54911234567890"
            ]
            
            for number in test_numbers:
                normalized = whatsapp_handler.normalize_phone_number(number)
                print(f"   {number} -> {normalized}")
            
            # Test con número específico
            test_phone = "+541130897333"
            normalized_test = whatsapp_handler.normalize_phone_number(test_phone)
            self.log_success(f"Número de prueba normalizado: {test_phone} -> {normalized_test}")
            
            # Validar que la normalización funciona
            if normalized_test and len(normalized_test) >= 10:
                self.log_success("Normalización de números funcional")
            else:
                self.log_error("Error en normalización de números")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"Error en integración de WhatsApp: {str(e)}")
            return False 