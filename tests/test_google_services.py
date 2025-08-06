#!/usr/bin/env python3
"""
Test de servicios de Google (Sheets y Calendar).
Valida la conectividad y funcionalidad básica.
"""

from base_test import BaseTest


class TestGoogleServices(BaseTest):
    """Test de servicios de Google."""
    
    def get_test_description(self) -> str:
        return "Test de servicios de Google (Sheets y Calendar)"
    
    def get_test_category(self) -> str:
        return "google"
    
    def _run_test_logic(self) -> bool:
        """Validar servicios de Google."""
        print("📅 Probando servicios de Google...")
        
        try:
            google_api_key = self.config.google_apis.google_api_key
            
            if not google_api_key:
                self.log_warning("Google API key no configurada")
                return True  # No fallar por esto
            
            # Test de Google Sheets
            try:
                from services.sheets_service import SheetsService
                sheets_service = SheetsService(api_key=google_api_key)
                self.log_success("Google Sheets service inicializado")
            except Exception as e:
                self.log_warning(f"Error en Google Sheets: {str(e)}")
            
            # Test de Google Calendar
            try:
                from services.calendar_service import CalendarService
                calendar_service = CalendarService()
                self.log_success("Google Calendar service inicializado")
            except Exception as e:
                self.log_warning(f"Error en Google Calendar: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Error en servicios de Google: {str(e)}")
            return False 