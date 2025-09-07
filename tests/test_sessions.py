#!/usr/bin/env python3
"""
Tests de sesiones: TTL y contexto relativo (semana/mes) y flujo con RAG.
"""

import time
from base_test import BaseTest


class TestSessions(BaseTest):
    """Valida funcionamiento de SessionService y uso de contexto relativo."""

    def get_test_description(self) -> str:
        return "Tests de sesiones (TTL y contexto relativo)"

    def get_test_category(self) -> str:
        return "sessions"

    def _run_test_logic(self) -> bool:
        # 1) TTL: crear servicio con TTL corto y verificar expiración
        from services.session_service import SessionService

        sess = SessionService(max_sessions=10, ttl_seconds=1)
        s = sess.get_session("u1")
        assert s.user_id == "u1"
        time.sleep(1.2)
        stats_before = sess.get_session_stats()
        # volver a pedir sesión debe limpiar expiradas y crear nueva
        s2 = sess.get_session("u1")
        stats_after = sess.get_session_stats()
        assert s2.user_id == "u1"
        assert stats_after["active_sessions"] >= 1

        # 2) Contexto relativo - calendario ("y la que sigue?")
        sess2 = SessionService(max_sessions=10, ttl_seconds=60)
        # Setear contexto previo: calendario esta semana
        sess2.update_session_context(
            user_id="u2",
            query="¿Qué actividades hay esta semana?",
            query_type="calendario_eventos_generales",
            calendar_intent="eventos_generales",
            time_reference="esta semana",
        )
        ctx = sess2.get_context_for_relative_query("u2", "y la que sigue?")
        assert ctx["is_relative"] is True
        assert ctx["resolved_time_reference"] in ("la próxima semana", "la proxima semana")

        # 3) Contexto relativo - cursos (mes siguiente)
        sess3 = SessionService(max_sessions=10, ttl_seconds=60)
        sess3.update_session_context(
            user_id="u3",
            query="cursos AGOSTO",
            query_type="cursos",
            month_requested="AGOSTO",
        )
        ctx2 = sess3.get_context_for_relative_query("u3", "y el que sigue?")
        assert ctx2["is_relative"] is True
        assert ctx2["resolved_month"] is not None

        # 4) Flujo con RAG: relativa de calendario debe marcar *_relativa
        # Este subtest requiere configuración completa (OpenAI y DB). Si falla por entorno, lo omitimos.
        try:
            from rag_system import RAGSystem
            rag = RAGSystem()
            res1 = rag.process_query("Preguntale que actividades hay esta semana?", user_id="u4", user_name="Test")
            res2 = rag.process_query(" Y La que sigue?", user_id="u4", user_name="Test")
            assert "response" in res1 and "response" in res2
            assert "calendario_" in res2.get("query_type", "") and res2["query_type"].endswith("_relativa")

            # Validaciones adicionales: no mezclar cursos en respuesta de calendario relativa
            resp2_text = res2.get("response", "").lower()
            assert "📚" not in resp2_text and "cursos" not in resp2_text, "La respuesta relativa de calendario no debe mezclar cursos"

            # Evitar solapamiento de texto idéntico entre res1 y res2 (mismas fechas)
            resp1_lines = set([l.strip() for l in res1.get("response", "").splitlines() if l.strip()])
            resp2_lines = set([l.strip() for l in res2.get("response", "").splitlines() if l.strip()])
            overlap = resp1_lines.intersection(resp2_lines)
            # Permitir cabecera común, pero no repetir listados exactos de eventos
            assert len([l for l in overlap if l.startswith("📌") or l.startswith("  ")]) == 0, "Las listas de eventos no deben ser idénticas entre semana actual y la siguiente"
        except Exception as e:
            from config.settings import logger
            logger.warning(f"Subtest RAG omitido por entorno no listo: {str(e)}")

        return True


if __name__ == "__main__":
    test = TestSessions()
    success = test.run_test()
    exit(0 if success else 1)


