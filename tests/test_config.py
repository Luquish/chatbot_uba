#!/usr/bin/env python3
"""
Test de configuración del sistema Chatbot UBA.
Valida que todas las variables de entorno estén configuradas correctamente.
"""

from base_test import BaseTest


class TestConfig(BaseTest):
    """Test de configuración del sistema."""
    
    def get_test_description(self) -> str:
        return "Validación de configuración completa del sistema"
    
    def get_test_category(self) -> str:
        return "config"
    
    def _run_test_logic(self) -> bool:
        """Validar configuración del sistema."""
        print("🔧 Probando configuración completa del sistema...")
        
        try:
            # Verificar OpenAI
            if not self.config.openai.openai_api_key:
                self.log_error("OpenAI API key no configurada")
                return False
            self.log_success("OpenAI configurado: True")
            self.log_info(f"Modelo principal: {self.config.openai.primary_model}")
            self.log_info(f"Modelo de embeddings: {self.config.openai.embedding_model}")
            
            # Verificar WhatsApp
            whatsapp_configured = all([
                self.config.whatsapp.whatsapp_api_token,
                self.config.whatsapp.whatsapp_phone_number_id,
                self.config.whatsapp.whatsapp_business_account_id
            ])
            self.log_success(f"WhatsApp configurado: {whatsapp_configured}")
            
            # Verificar base de datos
            db_configured = all([
                self.config.cloudsql.db_user,
                self.config.cloudsql.db_pass,
                self.config.cloudsql.db_name
            ])
            self.log_success(f"Base de datos configurada: {db_configured}")
            
            # Verificar Google APIs
            google_configured = bool(self.config.google_apis.google_api_key)
            self.log_success(f"Google APIs configuradas: {google_configured}")
            
            # Verificar configuración general
            if not all([whatsapp_configured, db_configured, google_configured]):
                self.log_warning("Algunas configuraciones están incompletas")
                return False
                
            self.log_success("Configuración cargada exitosamente")
            return True
            
        except Exception as e:
            self.log_error(f"Error en configuración: {str(e)}")
            return False 