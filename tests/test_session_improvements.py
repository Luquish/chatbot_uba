#!/usr/bin/env python3
"""
Test específico para validar las mejoras implementadas en el servicio de sesiones:
1. Refactorización de la lógica de consultas relativas
2. Singleton mejorado para testabilidad
3. Procesadores modulares de consultas relativas
"""

from base_test import BaseTest
import time


class TestSessionImprovements(BaseTest):
    """Test que valida las mejoras implementadas en el SessionService."""

    def get_test_description(self) -> str:
        return "Validación de mejoras en SessionService (procesadores modulares, singleton)"

    def get_test_category(self) -> str:
        return "sessions"

    def _run_test_logic(self) -> bool:
        """Valida las mejoras implementadas."""
        
        self.log_info("=== FASE 1: Test de procesadores modulares ===")
        
        # Test directo de los procesadores de consultas relativas
        from services.relative_query_processors import CourseRelativeProcessor, CalendarRelativeProcessor, RelativeQueryManager
        from services.session_service import UserSession
        
        # Test del procesador de cursos
        course_processor = CourseRelativeProcessor()
        
        # Crear sesión mock con contexto de cursos
        session_cursos = UserSession(user_id="test_user")
        session_cursos.last_query_type = "cursos"
        session_cursos.last_month_requested = "AGOSTO"
        
        assert course_processor.can_process(session_cursos), "Procesador de cursos debe poder procesar sesión con contexto de cursos"
        
        # Test de consulta relativa de cursos
        context = course_processor.process(session_cursos, "y el que sigue?", "test_user")
        assert context is not None, "Procesador debe detectar consulta relativa de cursos"
        assert context.is_relative, "Contexto debe marcar como relativo"
        assert context.resolved_month == "SEPTIEMBRE", f"Debe resolver AGOSTO + 1 = SEPTIEMBRE, obtuvo: {context.resolved_month}"
        
        self.log_success("✅ Procesador de cursos funcionando correctamente")
        
        # Test del procesador de calendario
        calendar_processor = CalendarRelativeProcessor()
        
        # Crear sesión mock con contexto de calendario
        session_calendar = UserSession(user_id="test_user")
        session_calendar.last_query_type = "calendario_eventos_generales"
        session_calendar.last_time_reference = "esta semana"
        session_calendar.last_calendar_intent = "eventos_generales"
        
        assert calendar_processor.can_process(session_calendar), "Procesador de calendario debe poder procesar sesión con contexto de calendario"
        
        # Test de consulta relativa de calendario
        context = calendar_processor.process(session_calendar, "y la que sigue?", "test_user")
        assert context is not None, "Procesador debe detectar consulta relativa de calendario"
        assert context.is_relative, "Contexto debe marcar como relativo"
        assert "próxima semana" in context.resolved_time_reference, f"Debe resolver como próxima semana, obtuvo: {context.resolved_time_reference}"
        
        self.log_success("✅ Procesador de calendario funcionando correctamente")
        
        # Test del gestor conjunto
        manager = RelativeQueryManager()
        
        # Test con sesión de cursos
        context_dict = manager.get_context_for_relative_query(session_cursos, "y el anterior?", "test_user")
        assert context_dict["is_relative"], "Manager debe detectar consulta relativa"
        assert context_dict["resolved_month"] == "JULIO", f"Debe resolver AGOSTO - 1 = JULIO, obtuvo: {context_dict['resolved_month']}"
        
        # Test con sesión de calendario
        context_dict = manager.get_context_for_relative_query(session_calendar, "y la anterior?", "test_user")
        assert context_dict["is_relative"], "Manager debe detectar consulta relativa de calendario"
        assert "semana pasada" in context_dict["resolved_time_reference"], f"Debe resolver como semana pasada, obtuvo: {context_dict['resolved_time_reference']}"
        
        self.log_success("✅ Gestor de consultas relativas funcionando correctamente")
        
        self.log_info("=== FASE 2: Test de singleton mejorado ===")
        
        from services.session_service import SessionServiceSingleton, SessionService, get_session_service
        
        # Resetear singleton para test limpio
        SessionServiceSingleton.reset_instance()
        
        # Test de obtención de instancia con parámetros personalizados
        service1 = get_session_service(max_sessions=100, ttl_seconds=300)
        service2 = get_session_service(max_sessions=200, ttl_seconds=600)  # Estos parámetros deben ser ignorados
        
        assert service1 is service2, "Singleton debe retornar la misma instancia"
        assert service1.max_sessions == 100, f"Debe usar parámetros de la primera inicialización, obtuvo: {service1.max_sessions}"
        assert service1.ttl_seconds == 300, f"Debe usar TTL de la primera inicialización, obtuvo: {service1.ttl_seconds}"
        
        self.log_success("✅ Singleton funcionando correctamente con parámetros")
        
        # Test de reset de instancia
        SessionServiceSingleton.reset_instance()
        service3 = get_session_service(max_sessions=50)
        
        assert service3 is not service1, "Después del reset debe crear nueva instancia"
        assert service3.max_sessions == 50, "Nueva instancia debe usar nuevos parámetros"
        
        self.log_success("✅ Reset de singleton funcionando correctamente")
        
        # Test de instancia personalizada
        custom_service = SessionService(max_sessions=999, ttl_seconds=999, enable_background_sweeper=False)
        SessionServiceSingleton.set_instance(custom_service)
        
        service4 = get_session_service()
        assert service4 is custom_service, "Debe usar la instancia personalizada establecida"
        assert service4.max_sessions == 999, "Debe mantener configuración de la instancia personalizada"
        
        self.log_success("✅ Instancia personalizada funcionando correctamente")
        
        self.log_info("=== FASE 3: Test de integración completa ===")
        
        # Crear un nuevo servicio para test de integración
        SessionServiceSingleton.reset_instance()
        service = get_session_service(max_sessions=10, ttl_seconds=60, enable_background_sweeper=False)
        
        # Test de flujo completo con procesadores modulares
        user_id = "integration_test_user"
        
        # 1) Establecer contexto de cursos
        service.update_session_context(
            user_id=user_id,
            query="cursos de MAYO",
            query_type="cursos",
            month_requested="MAYO"
        )
        
        # 2) Realizar consulta relativa
        context = service.get_context_for_relative_query(user_id, "y el siguiente?")
        
        assert context["is_relative"], "Debe detectar consulta relativa en integración"
        assert context["resolved_month"] == "JUNIO", f"Debe resolver MAYO + 1 = JUNIO, obtuvo: {context['resolved_month']}"
        assert context["query_type"] == "cursos", "Debe mantener tipo de consulta de cursos"
        
        self.log_success("✅ Integración completa funcionando correctamente")
        
        # 3) Cambiar a contexto de calendario
        service.update_session_context(
            user_id=user_id,
            query="eventos este mes",
            query_type="calendario_eventos_generales",
            calendar_intent="eventos_generales",
            time_reference="este mes"
        )
        
        # 4) Consulta relativa de calendario
        context = service.get_context_for_relative_query(user_id, "y el que viene?")
        
        assert context["is_relative"], "Debe detectar consulta relativa de calendario en integración"
        assert context["query_type"] == "calendario_eventos_generales", "Debe mantener tipo de consulta de calendario"
        assert "próximo mes" in context["resolved_time_reference"], f"Debe resolver como próximo mes, obtuvo: {context['resolved_time_reference']}"
        
        self.log_success("✅ Cambio de contexto funcionando correctamente")
        
        # Limpiar
        SessionServiceSingleton.reset_instance()
        
        self.log_info("=== RESUMEN DE MEJORAS VALIDADAS ===")
        self.log_success("🎯 Todas las mejoras implementadas están funcionando:")
        self.log_success("   1. ✅ Procesadores modulares de consultas relativas")
        self.log_success("   2. ✅ Singleton mejorado con testabilidad")
        self.log_success("   3. ✅ Integración completa y cambio de contexto")
        self.log_success("   4. ✅ Lógica refactorizada manteniendo funcionalidad")
        
        return True


if __name__ == "__main__":
    test = TestSessionImprovements()
    success = test.run_test()
    exit(0 if success else 1)