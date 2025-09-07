#!/usr/bin/env python3
"""
Test de diagnóstico para identificar por qué las sesiones no funcionan en producción.
Este test simula diferentes escenarios y proporciona información detallada de debugging.
"""

from base_test import BaseTest
import json


class TestSessionDiagnostic(BaseTest):
    """Test de diagnóstico para resolver problemas de producción."""

    def get_test_description(self) -> str:
        return "Diagnóstico completo de sesiones para resolver problemas de producción"

    def get_test_category(self) -> str:
        return "diagnostic"

    def _run_test_logic(self) -> bool:
        """Ejecuta diagnósticos completos."""
        
        self.log_info("=== DIAGNÓSTICO 1: Verificar componentes básicos ===")
        
        try:
            from services.session_service import get_session_service, SessionServiceSingleton
            from services.relative_query_processors import RelativeQueryManager
            
            # Reset para test limpio
            SessionServiceSingleton.reset_instance()
            service = get_session_service(max_sessions=10, ttl_seconds=300, enable_background_sweeper=False)
            
            self.log_success("✅ Componentes de sesión se importan correctamente")
            
            # Verificar configuración
            self.log_info(f"📊 Configuración del servicio:")
            self.log_info(f"   - Max sesiones: {service.max_sessions}")
            self.log_info(f"   - TTL: {service.ttl_seconds}s")
            self.log_info(f"   - Sweeper activo: {service._sweeper_thread is not None}")
            
        except Exception as e:
            self.log_error(f"❌ Error en importación de componentes: {e}")
            return False
        
        self.log_info("=== DIAGNÓSTICO 2: Flujo de calendario paso a paso ===")
        
        user_id = "diagnostic_user_cal"
        
        try:
            # 1. Crear sesión inicial
            session = service.get_session(user_id)
            self.log_info(f"📝 Sesión creada: user_id={session.user_id}")
            
            # 2. Simular primera consulta de calendario
            service.update_session_context(
                user_id=user_id,
                query="¿Qué actividades hay esta semana?",
                query_type="calendario_eventos_generales",
                calendar_intent="eventos_generales",
                time_reference="esta semana"
            )
            
            # 3. Verificar que el contexto se guardó
            updated_session = service.get_session(user_id)
            self.log_info(f"📝 Contexto después de primera consulta:")
            self.log_info(f"   - Query type: '{updated_session.last_query_type}'")
            self.log_info(f"   - Calendar intent: '{updated_session.last_calendar_intent}'")
            self.log_info(f"   - Time reference: '{updated_session.last_time_reference}'")
            self.log_info(f"   - Last query: '{updated_session.last_query}'")
            
            # 4. Probar consulta relativa
            context = service.get_context_for_relative_query(user_id, "¿Y la que sigue?")
            self.log_info(f"📝 Contexto relativo generado:")
            self.log_info(f"   - Es relativa: {context.get('is_relative')}")
            self.log_info(f"   - Time reference resuelto: '{context.get('resolved_time_reference')}'")
            self.log_info(f"   - Query type: '{context.get('query_type')}'")
            self.log_info(f"   - Calendar intent: '{context.get('calendar_intent')}'")
            self.log_info(f"   - Explicación: '{context.get('explanation')}'")
            
            if context.get('is_relative'):
                self.log_success("✅ Consulta relativa de calendario detectada correctamente")
            else:
                self.log_warning("⚠️  Consulta relativa NO detectada - posible problema")
                
        except Exception as e:
            self.log_error(f"❌ Error en flujo de calendario: {e}")
            import traceback
            self.log_error(f"   Traceback: {traceback.format_exc()}")
            return False
        
        self.log_info("=== DIAGNÓSTICO 3: Flujo de cursos paso a paso ===")
        
        user_id_cursos = "diagnostic_user_cursos"
        
        try:
            # 1. Consulta inicial de cursos
            service.update_session_context(
                user_id=user_id_cursos,
                query="¿Qué cursos hay en agosto?",
                query_type="cursos",
                month_requested="AGOSTO"
            )
            
            session_cursos = service.get_session(user_id_cursos)
            self.log_info(f"📝 Contexto de cursos:")
            self.log_info(f"   - Query type: '{session_cursos.last_query_type}'")
            self.log_info(f"   - Month requested: '{session_cursos.last_month_requested}'")
            
            # 2. Consulta relativa de cursos
            context = service.get_context_for_relative_query(user_id_cursos, "¿Y el que sigue?")
            self.log_info(f"📝 Contexto relativo de cursos:")
            self.log_info(f"   - Es relativa: {context.get('is_relative')}")
            self.log_info(f"   - Mes resuelto: '{context.get('resolved_month')}'")
            self.log_info(f"   - Explicación: '{context.get('explanation')}'")
            
            if context.get('is_relative') and context.get('resolved_month') == 'SEPTIEMBRE':
                self.log_success("✅ Consulta relativa de cursos funciona correctamente")
            else:
                self.log_warning("⚠️  Problema en consulta relativa de cursos")
                
        except Exception as e:
            self.log_error(f"❌ Error en flujo de cursos: {e}")
            return False
        
        self.log_info("=== DIAGNÓSTICO 4: Test con RAG System ===")
        
        try:
            # Intentar usar el RAG system real
            from rag_system import RAGSystem
            rag = RAGSystem()
            
            user_id_rag = "diagnostic_rag_user"
            
            self.log_info("📝 Probando flujo con RAG System real...")
            
            # Primera consulta
            response1 = rag.process_query("¿Qué actividades hay esta semana?", user_id=user_id_rag, user_name="DiagnosticUser")
            
            self.log_info(f"📝 Respuesta RAG 1:")
            self.log_info(f"   - Query type: '{response1.get('query_type')}'")
            self.log_info(f"   - Has response: {len(response1.get('response', '')) > 0}")
            self.log_info(f"   - Response preview: '{response1.get('response', '')[:100]}...'")
            
            # Verificar sesión después de RAG
            rag_session = service.get_session(user_id_rag)
            self.log_info(f"📝 Sesión después de RAG:")
            self.log_info(f"   - Query type: '{rag_session.last_query_type}'")
            self.log_info(f"   - Time reference: '{rag_session.last_time_reference}'")
            
            # Segunda consulta relativa
            response2 = rag.process_query("¿Y la que sigue?", user_id=user_id_rag, user_name="DiagnosticUser")
            
            self.log_info(f"📝 Respuesta RAG 2 (relativa):")
            self.log_info(f"   - Query type: '{response2.get('query_type')}'")
            self.log_info(f"   - Has response: {len(response2.get('response', '')) > 0}")
            self.log_info(f"   - Es relativa: {'_relativa' in response2.get('query_type', '')}")
            
            if '_relativa' in response2.get('query_type', ''):
                self.log_success("✅ RAG System detecta correctamente consultas relativas")
            else:
                self.log_warning("⚠️  RAG System NO detecta consultas relativas - revisar integración")
                
        except ImportError:
            self.log_warning("⚠️  RAG System no disponible - test omitido")
        except Exception as e:
            self.log_error(f"❌ Error con RAG System: {e}")
            self.log_info("   Esto podría explicar por qué no funciona en producción")
        
        self.log_info("=== DIAGNÓSTICO 5: Verificar patrones de consulta ===")
        
        # Test de diferentes variaciones de consultas relativas
        test_queries = [
            "y la que sigue?",
            "Y la que sigue?",
            "¿Y la que sigue?",
            "y el que sigue?",
            "¿Y el que sigue?",
            "y el siguiente?",
            "y la siguiente?",
            "y el anterior?",
            "y la anterior?"
        ]
        
        # Test con contexto de calendario
        service.update_session_context(
            user_id="pattern_test",
            query="eventos esta semana",
            query_type="calendario_eventos_generales",
            calendar_intent="eventos_generales", 
            time_reference="esta semana"
        )
        
        detected_count = 0
        for query in test_queries:
            context = service.get_context_for_relative_query("pattern_test", query)
            if context.get('is_relative'):
                detected_count += 1
                self.log_success(f"✅ Patrón detectado: '{query}'")
            else:
                self.log_warning(f"⚠️  Patrón NO detectado: '{query}'")
        
        self.log_info(f"📊 Patrones detectados: {detected_count}/{len(test_queries)}")
        
        self.log_info("=== RESUMEN DE DIAGNÓSTICO ===")
        
        stats = service.get_session_stats()
        self.log_info(f"📊 Estadísticas finales:")
        self.log_info(f"   - Sesiones activas: {stats['active_sessions']}")
        self.log_info(f"   - Max sesiones: {stats['max_sessions']}")
        self.log_info(f"   - TTL configurado: {stats['ttl_seconds']}s")
        
        self.log_success("🔍 Diagnóstico completado - revisa los logs para identificar problemas")
        
        return True


if __name__ == "__main__":
    test = TestSessionDiagnostic()
    success = test.run_test()
    exit(0 if success else 1)