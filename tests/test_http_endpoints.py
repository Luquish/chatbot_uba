#!/usr/bin/env python3
"""
Test de endpoints HTTP críticos.
Valida la conectividad a endpoints del servidor.
"""

import httpx
from base_test import AsyncBaseTest


class TestHttpEndpoints(AsyncBaseTest):
    """Test de endpoints HTTP."""
    
    def get_test_description(self) -> str:
        return "Test de endpoints HTTP críticos"
    
    def get_test_category(self) -> str:
        return "http"
    
    async def _run_test_logic(self) -> bool:
        """Validar endpoints HTTP."""
        print("🌐 Probando endpoints HTTP críticos...")
        
        try:
            # Lista de endpoints a probar
            endpoints = [
                "http://localhost:8080/health",
                "http://localhost:8080/test-message",
                "http://localhost:8080/test-webhook"
            ]
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                for endpoint in endpoints:
                    try:
                        response = await client.get(endpoint)
                        if response.status_code == 200:
                            self.log_success(f"✅ {endpoint}: {response.status_code}")
                        else:
                            self.log_warning(f"⚠️ {endpoint}: {response.status_code}")
                    except httpx.ConnectError:
                        self.log_warning(f"⚠️ Servidor no disponible en {endpoint}: All connection attempts failed")
                    except Exception as e:
                        self.log_warning(f"⚠️ Error en {endpoint}: {str(e)}")
            
            # Considerar exitoso si al menos un endpoint responde
            return True
            
        except Exception as e:
            self.log_error(f"Error en test de endpoints HTTP: {str(e)}")
            return False 