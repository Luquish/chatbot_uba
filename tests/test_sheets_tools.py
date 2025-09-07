#!/usr/bin/env python3
"""
Tests para herramientas de Sheets nuevas con cobertura de normalización en utils.
"""

from base_test import BaseTest
from utils.text_utils import normalize_text, detect_day_token, extract_catedra_number, is_count_catedras_query
from utils.date_utils import DateUtils


class TestSheetsTools(BaseTest):
    """Valida utilidades de normalización y herramientas de sheets a alto nivel."""

    def get_test_description(self) -> str:
        return "Tests de normalización (text/date utils) con escenarios de sheets"

    def get_test_category(self) -> str:
        return "sheets_utils"

    def _run_test_logic(self) -> bool:
        # Text utils
        assert normalize_text("  ¡Hólá, Mundo!  ") == "hola mundo"
        assert extract_catedra_number("Cátedra 2 de Anatomía") == "2"
        assert is_count_catedras_query("¿Cuantas catedras de anatomia hay?") is True
        assert detect_day_token("¿Atienden el Miércoles?") == 'W'

        # Date utils
        du = DateUtils()
        assert DateUtils.month_name_to_num("mayo") == 5
        assert DateUtils.num_to_month_name(8, uppercase=True) == "AGOSTO"
        assert DateUtils.detect_month_from_text("Cursada en octubre") == "octubre"
        assert DateUtils.get_weekday_abbr("miércoles") == "mié"
        assert DateUtils.get_weekday_abbr("miercoles") == "mié"

        # Sanity: instanciar herramientas si están disponibles (sin llamadas externas)
        try:
            from services.tools.horarios_catedra_tool import HorariosCatedraTool
            from services.tools.horarios_lic_tec_tool import HorariosLicTecTool
            from services.tools.horarios_secretarias_tool import HorariosSecretariasTool
            from services.tools.mails_nuevo_espacio_tool import MailsNuevoEspacioTool
            tools = [
                HorariosCatedraTool(None),
                HorariosLicTecTool(None),
                HorariosSecretariasTool(None),
                MailsNuevoEspacioTool(None)
            ]
            for t in tools:
                assert hasattr(t, 'can_handle') and hasattr(t, 'execute')
        except Exception as e:
            self.log_warning(f"Instanciación de herramientas sin servicio: {e}")

        return True


if __name__ == "__main__":
    test = TestSheetsTools()
    success = test.run_test()
    exit(0 if success else 1)


