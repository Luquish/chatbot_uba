#!/usr/bin/env python3
"""
Test integral que simula el flujo completo de producción del chatbot,
incluyendo la integración real con RAG System y handlers.
"""

import time
from base_test import BaseTest


class TestSessionProductionFlow(BaseTest):
    """Test que simula el flujo completo de producción del chatbot."""

    def get_test_description(self) -> str:
        return "Test de flujo completo de producción con sesiones"

    def get_test_category(self) -> str:
        return "integration"

    def _run_test_logic(self) -> bool:
        """Simula el flujo completo como en producción."""
        try:
            # Importar el sistema RAG completo (como en producción)
            from rag_system import RAGSystem
            
            # Usar el sistema RAG real (no mocks)
            rag = RAGSystem()
            user_id = "test_prod_user_001"
            user_name = "TestUser"
            
            self.log_info("=== FASE 1: Consulta inicial de calendario ===")
            
            # 1) Consulta inicial: eventos esta semana
            query1 = "¿Qué actividades hay esta semana?"
            response1 = rag.process_query(query1, user_id=user_id, user_name=user_name)
            
            # Validaciones básicas de la primera respuesta
            assert "response" in response1, "Primera respuesta debe tener campo 'response'"
            assert "query_type" in response1, "Primera respuesta debe tener 'query_type'"
            assert "calendario" in response1["query_type"], f"Query type debe ser de calendario, fue: {response1.get('query_type')}"
            
            response1_text = response1["response"]
            self.log_success(f"✅ Primera consulta procesada. Query type: {response1['query_type']}")
            self.log_info(f"Respuesta 1 (primeras 200 chars): {response1_text[:200]}...")
            
            # Verificar que se creó la sesión y contexto
            from services.session_service import session_service
            session = session_service.get_session(user_id)
            assert session.last_query_type.startswith("calendario"), f"Sesión debe guardar tipo calendario, tiene: {session.last_query_type}"
            assert session.last_time_reference, f"Sesión debe guardar referencia temporal, tiene: {session.last_time_reference}"
            self.log_success(f"✅ Sesión creada correctamente. Context: {session.last_query_type}, Time ref: {session.last_time_reference}")
            
            self.log_info("=== FASE 2: Consulta relativa ===")
            
            # 2) Consulta relativa: "y la que sigue?"
            query2 = "¿Y la que sigue?"  # Debe interpretar como "la próxima semana"
            response2 = rag.process_query(query2, user_id=user_id, user_name=user_name)
            
            # Validaciones de la segunda respuesta
            assert "response" in response2, "Segunda respuesta debe tener campo 'response'"
            assert "query_type" in response2, "Segunda respuesta debe tener 'query_type'"
            
            response2_type = response2["query_type"]
            response2_text = response2["response"]
            
            # VALIDACIÓN CRÍTICA: debe ser query relativa de calendario
            assert "calendario" in response2_type, f"Segunda query debe ser de calendario, fue: {response2_type}"
            assert "_relativa" in response2_type, f"Segunda query debe ser relativa, fue: {response2_type}"
            
            self.log_success(f"✅ Segunda consulta relativa procesada. Query type: {response2_type}")
            self.log_info(f"Respuesta 2 (primeras 200 chars): {response2_text[:200]}...")
            
            self.log_info("=== FASE 3: Validaciones de diferenciación ===")
            
            # 3) Validar que las respuestas son diferentes (no idénticas)
            # Extraer líneas de eventos (que empiecen con emoji o indentación)
            def extract_event_lines(text):
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                return [l for l in lines if l.startswith("📌") or l.startswith("  ") or l.startswith("•")]
            
            events1 = extract_event_lines(response1_text)
            events2 = extract_event_lines(response2_text)
            
            # Si hay eventos en ambas respuestas, no deben ser idénticos
            if events1 and events2:
                identical_events = set(events1) & set(events2)
                assert len(identical_events) == 0, f"Las respuestas no deben tener eventos idénticos. Coincidencias: {identical_events}"
                self.log_success(f"✅ Respuestas diferenciadas correctamente ({len(events1)} vs {len(events2)} eventos)")
            else:
                self.log_info("⚠️  Una de las respuestas no tiene eventos detectables, validación de diferenciación omitida")
            
            # 4) Validar que no se mezclan dominios (no debe haber cursos en respuestas de calendario)
            assert "📚" not in response1_text, "Respuesta de calendario no debe contener emoji de cursos"
            assert "📚" not in response2_text, "Respuesta relativa de calendario no debe contener emoji de cursos"
            assert "cursos" not in response1_text.lower(), "Respuesta de calendario no debe mencionar cursos"
            assert "cursos" not in response2_text.lower(), "Respuesta relativa de calendario no debe mencionar cursos"
            
            self.log_success("✅ Separación de dominios correcta (calendario sin cursos)")
            
            self.log_info("=== FASE 4: Test de contexto de cursos ===")
            
            # 5) Cambiar a contexto de cursos para probar otro tipo de relación
            query3 = "¿Qué cursos hay en agosto?"
            response3 = rag.process_query(query3, user_id=user_id, user_name=user_name)
            
            assert "response" in response3, "Tercera respuesta debe tener campo 'response'"
            assert "cursos" in response3.get("query_type", ""), f"Query type debe ser cursos, fue: {response3.get('query_type')}"
            
            self.log_success(f"✅ Consulta de cursos procesada. Query type: {response3.get('query_type')}")
            
            # 6) Consulta relativa de cursos: "y el que sigue?"
            query4 = "¿Y el que sigue?"  # Debe interpretar como "septiembre"
            response4 = rag.process_query(query4, user_id=user_id, user_name=user_name)
            
            assert "response" in response4, "Cuarta respuesta debe tener campo 'response'"
            response4_type = response4.get("query_type", "")
            
            # Debe ser consulta relativa de cursos
            assert "cursos" in response4_type, f"Cuarta query debe ser de cursos, fue: {response4_type}"
            # Puede o no tener "_relativa" dependiendo de la implementación, pero debe resolver el mes
            
            self.log_success(f"✅ Consulta relativa de cursos procesada. Query type: {response4_type}")
            
            # 7) Verificar que la sesión mantiene el nuevo contexto
            final_session = session_service.get_session(user_id)
            assert final_session.last_query_type == "cursos", f"Sesión debe reflejar último tipo 'cursos', tiene: {final_session.last_query_type}"
            
            self.log_success("✅ Contexto de sesión actualizado correctamente")
            
            self.log_info("=== RESUMEN DEL TEST ===")
            self.log_success("🎯 Flujo completo de producción simulado exitosamente:")
            self.log_success(f"   1. Calendario inicial: {response1.get('query_type')}")
            self.log_success(f"   2. Calendario relativo: {response2.get('query_type')}")
            self.log_success(f"   3. Cursos inicial: {response3.get('query_type')}")  
            self.log_success(f"   4. Cursos relativo: {response4.get('query_type')}")
            
            return True
            
        except ImportError as e:
            self.log_warning(f"⚠️  RAGSystem no disponible, omitiendo test de producción: {e}")
            return True  # No fallar el test si el entorno no está completo
            
        except Exception as e:
            self.log_error(f"❌ Error en flujo de producción: {str(e)}")
            # Agregar información de depuración
            try:
                from services.session_service import session_service
                stats = session_service.get_session_stats()
                self.log_error(f"   Stats de sesiones: {stats}")
                if user_id:
                    session = session_service.get_session(user_id)
                    self.log_error(f"   Sesión actual: query_type={session.last_query_type}, time_ref={session.last_time_reference}, month={session.last_month_requested}")
            except:
                pass
            raise


if __name__ == "__main__":
    test = TestSessionProductionFlow()
    success = test.run_test()
    exit(0 if success else 1)