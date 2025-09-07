#!/usr/bin/env python3
"""
Clase base para todos los tests del Chatbot UBA.
Proporciona configuración común y utilidades compartidas.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Añadir el directorio raíz al path para importar módulos
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importar configuración
from config.settings import config


class BaseTest:
    """Clase base para todos los tests del sistema."""
    
    def __init__(self):
        """Inicializar test base."""
        self.config = config
        self.passed = False
        self.error_message = ""
        self.details = {}
        
    def log_info(self, message: str):
        """Log de información."""
        logging.info(f"   ℹ️ {message}")
        
    def log_success(self, message: str):
        """Log de éxito."""
        logging.info(f"   ✅ {message}")
        
    def log_warning(self, message: str):
        """Log de advertencia."""
        logging.warning(f"   ⚠️ {message}")
        
    def log_error(self, message: str):
        """Log de error."""
        logging.error(f"   ❌ {message}")
        
    def log_step(self, message: str):
        """Log de paso."""
        logging.info(f"   🔧 {message}")
        
    def run_test(self) -> Dict[str, Any]:
        """
        Ejecutar el test y retornar resultados.
        
        Returns:
            Dict con resultados del test
        """
        try:
            self.passed = self._run_test_logic()
            return {
                "name": self.__class__.__name__,
                "passed": self.passed,
                "error_message": self.error_message,
                "details": self.details
            }
        except Exception as e:
            self.passed = False
            self.error_message = str(e)
            return {
                "name": self.__class__.__name__,
                "passed": False,
                "error_message": str(e),
                "details": {}
            }
    
    def _run_test_logic(self) -> bool:
        """
        Lógica específica del test. Debe ser implementada por subclases.
        
        Returns:
            bool: True si el test pasa, False en caso contrario
        """
        raise NotImplementedError("Subclases deben implementar _run_test_logic")
    
    def get_test_description(self) -> str:
        """Retorna descripción del test."""
        return "Test base - debe ser sobrescrito"
    
    def get_test_category(self) -> str:
        """Retorna categoría del test."""
        return "base"


class AsyncBaseTest(BaseTest):
    """Clase base para tests asíncronos."""
    
    async def run_test(self) -> Dict[str, Any]:
        """
        Ejecutar el test asíncrono y retornar resultados.
        
        Returns:
            Dict con resultados del test
        """
        try:
            self.passed = await self._run_test_logic()
            return {
                "name": self.__class__.__name__,
                "passed": self.passed,
                "error_message": self.error_message,
                "details": self.details
            }
        except Exception as e:
            self.passed = False
            self.error_message = str(e)
            return {
                "name": self.__class__.__name__,
                "passed": False,
                "error_message": str(e),
                "details": {}
            }
    
    async def _run_test_logic(self) -> bool:
        """
        Lógica específica del test asíncrono. Debe ser implementada por subclases.
        
        Returns:
            bool: True si el test pasa, False en caso contrario
        """
        raise NotImplementedError("Subclases deben implementar _run_test_logic") 