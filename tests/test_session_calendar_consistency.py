#!/usr/bin/env python3
"""
Test de consistencia de sesiones para calendario: valida que una consulta relativa
("y la que sigue?") no repita exactamente los mismos eventos de la primera respuesta
y que no mezcle cursos en respuestas de calendario.
"""

from base_test import BaseTest


class DummyCalendarService:
    """Servicio de calendario stub para pruebas determinísticas."""

    def get_events_this_week(self):
        return [
            {
                'summary': 'Reunión académica',
                'start': 'Lunes 1 de 10:00 hs a 11:00 hs',
                'end': None,
                'description': '',
                'same_day': True,
                'calendar_type': 'actividades_cecim',
            },
            {
                'summary': 'Charla de bienestar estudiantil',
                'start': 'Miércoles 3 de 14:00 hs a 15:00 hs',
                'end': None,
                'description': '',
                'same_day': True,
                'calendar_type': 'actividades_cecim',
            },
        ]

    def get_upcoming_events(self):
        # Usamos esta lista para simular "próxima semana" en este test
        return [
            {
                'summary': 'Taller de inserción universitaria',
                'start': 'Lunes 8 de 10:00 hs a 12:00 hs',
                'end': None,
                'description': '',
                'same_day': True,
                'calendar_type': 'actividades_cecim',
            },
            {
                'summary': 'Sesión informativa de becas',
                'start': 'Jueves 11 de 16:00 hs a 17:00 hs',
                'end': None,
                'description': '',
                'same_day': True,
                'calendar_type': 'actividades_cecim',
            },
        ]


class TestSessionCalendarConsistency(BaseTest):
    """Valida consistencia de respuesta entre semana actual y la siguiente."""

    def get_test_description(self) -> str:
        return "Sesiones: consistencia de calendario entre semana actual y siguiente"

    def get_test_category(self) -> str:
        return "sessions"

    def _run_test_logic(self) -> bool:
        from services.session_service import SessionService
        from handlers.calendar_handler import get_calendar_events

        # Preparar servicio de sesiones y calendario stub
        sess = SessionService(max_sessions=10, ttl_seconds=60)
        cal = DummyCalendarService()

        # Primera consulta: semana actual
        sess.update_session_context(
            user_id="u_cal", query="¿Qué actividades hay esta semana?",
            query_type="calendario_eventos_generales",
            calendar_intent="eventos_generales",
            time_reference="esta semana",
        )
        res1_text = get_calendar_events(cal, calendar_intent=None)  # Semana actual
        assert res1_text and "Eventos encontrados" in res1_text
        self.log_success("Respuesta de semana actual generada")

        # Segunda consulta relativa: "y la que sigue?"
        ctx = sess.get_context_for_relative_query("u_cal", "y la que sigue?")
        assert ctx["is_relative"] is True and ctx["resolved_time_reference"]
        # Simular respuesta para próxima semana (usamos get_upcoming_events)
        res2_text = get_calendar_events(cal, calendar_intent="eventos_generales")
        assert res2_text and "Eventos encontrados" in res2_text
        self.log_success("Respuesta de próxima semana generada")

        # Validar que no haya solapamiento exacto de líneas de eventos (excluyendo cabecera)
        res1_lines = [l.strip() for l in res1_text.splitlines() if l.strip()]
        res2_lines = [l.strip() for l in res2_text.splitlines() if l.strip()]
        # Filtrar cabeceras
        res1_ev = [l for l in res1_lines if l.startswith("📌") or l.startswith("  ")]
        res2_ev = [l for l in res2_lines if l.startswith("📌") or l.startswith("  ")]
        overlap = set(res1_ev).intersection(set(res2_ev))
        assert len(overlap) == 0, f"No debería haber eventos idénticos entre semana actual y próxima. Overlap: {overlap}"

        # Validar que no aparezca texto de cursos en respuestas de calendario
        assert "📚" not in res1_text and "cursos" not in res1_text.lower()
        assert "📚" not in res2_text and "cursos" not in res2_text.lower()

        return True


if __name__ == "__main__":
    test = TestSessionCalendarConsistency()
    success = test.run_test()
    exit(0 if success else 1)


