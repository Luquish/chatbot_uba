#!/usr/bin/env python3
"""
Tests específicos para validar que las excepciones de sesión se lancen correctamente.
Este test verifica que InvalidSessionDataError se lance con datos inválidos.
"""

import pytest
from base_test import BaseTest
from services.session_service import SessionService, InvalidSessionDataError, SessionError


class TestSessionExceptions(BaseTest):
    """Test que valida que las excepciones de sesión se lancen correctamente."""

    def get_test_description(self) -> str:
        return "Validación de excepciones de sesión (InvalidSessionDataError)"

    def get_test_category(self) -> str:
        return "sessions"

    def _run_test_logic(self) -> bool:
        """Valida que las excepciones se lancen correctamente."""
        
        self.log_info("=== TEST 1: Validación de user_id inválido ===")
        
        service = SessionService(max_sessions=10, ttl_seconds=60)
        
        # Test user_id vacío
        try:
            service.get_session("")
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con user_id vacío")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        # Test user_id None
        try:
            service.get_session(None)
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con user_id None")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        # Test user_id no string
        try:
            service.get_session(123)
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con user_id no string")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        # Test user_id solo espacios
        try:
            service.get_session("   ")
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con user_id solo espacios")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        self.log_info("=== TEST 2: Validación de query inválido en update_session_context ===")
        
        # Test query vacío
        try:
            service.update_session_context(
                user_id="test_user",
                query="",
                query_type="cursos"
            )
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con query vacío")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        # Test query None
        try:
            service.update_session_context(
                user_id="test_user",
                query=None,
                query_type="cursos"
            )
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con query None")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        # Test query no string
        try:
            service.update_session_context(
                user_id="test_user",
                query=123,
                query_type="cursos"
            )
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con query no string")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        self.log_info("=== TEST 3: Validación de query_type inválido ===")
        
        # Test query_type vacío
        try:
            service.update_session_context(
                user_id="test_user",
                query="¿Qué cursos hay?",
                query_type=""
            )
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con query_type vacío")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        # Test query_type None
        try:
            service.update_session_context(
                user_id="test_user",
                query="¿Qué cursos hay?",
                query_type=None
            )
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con query_type None")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        # Test query_type no string
        try:
            service.update_session_context(
                user_id="test_user",
                query="¿Qué cursos hay?",
                query_type=123
            )
            self.log_error("❌ Debería haber lanzado InvalidSessionDataError con query_type no string")
            return False
        except InvalidSessionDataError as e:
            self.log_success(f"✅ InvalidSessionDataError lanzada correctamente: {e}")
        except Exception as e:
            self.log_error(f"❌ Excepción incorrecta lanzada: {type(e).__name__}: {e}")
            return False
        
        self.log_info("=== TEST 4: Validación de casos válidos (no deben lanzar excepciones) ===")
        
        # Test casos válidos
        try:
            # user_id válido
            session = service.get_session("test_user_valid")
            assert session.user_id == "test_user_valid"
            self.log_success("✅ get_session con user_id válido funciona correctamente")
            
            # update_session_context válido
            service.update_session_context(
                user_id="test_user_valid",
                query="¿Qué cursos hay en agosto?",
                query_type="cursos",
                month_requested="AGOSTO"
            )
            self.log_success("✅ update_session_context con datos válidos funciona correctamente")
            
        except Exception as e:
            self.log_error(f"❌ Casos válidos lanzaron excepción inesperada: {type(e).__name__}: {e}")
            return False
        
        self.log_success("🎉 TODOS LOS TESTS DE EXCEPCIONES PASARON CORRECTAMENTE")
        return True


if __name__ == "__main__":
    test = TestSessionExceptions()
    success = test.run_test()
    exit(0 if success else 1)
