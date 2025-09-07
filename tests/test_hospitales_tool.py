"""
Tests para la herramienta de hospitales.
"""
from base_test import BaseTest
from services.tools.hospitales_tool import HospitalesTool


class TestHospitalesTool(BaseTest):
    """Tests para la herramienta de hospitales."""
    
    def get_test_description(self) -> str:
        return "Test de la herramienta de hospitales (ubicación, mapas, consultas específicas)"
    
    def get_test_category(self) -> str:
        return "tools"
    
    def _run_test_logic(self) -> bool:
        """Ejecutar tests de la herramienta de hospitales."""
        try:
            # Inicializar herramienta
            tool = HospitalesTool()
            
            # Test 1: Inicialización
            if tool.name != "hospitales":
                return False
            if tool.priority != 75:
                return False
            if not tool.mapa_url or "tinyurl.com" not in tool.mapa_url:
                return False
            
            # Test 2: Detección de consultas sobre hospitales
            test_queries = [
                "donde queda el hospital durand",
                "dónde está el hospital de clínicas",
                "ubicación del hospital italiano",
                "mapa de hospitales",
                "como llegar al hospital alemán",
                "udh ubicación"
            ]
            
            for query in test_queries:
                decision = tool.can_handle(query, {})
                if decision.score <= 0.0:
                    return False
                if not tool.accepts(decision.score):
                    return False
            
            # Test 3: Ejecución de consulta general
            result = tool.execute("donde estan los hospitales", {}, {})
            if not result.response or "Mapa de Hospitales" not in result.response:
                return False
            if "tinyurl.com" not in result.response:
                return False
            if result.sources != ["Mapa de Hospitales UDH-CECIM"]:
                return False
            
            # Test 4: Ejecución de consulta específica
            result = tool.execute("donde queda el hospital durand", {}, {})
            if not result.response or "Durand" not in result.response:
                return False
            if result.metadata.get('hospital_mentioned') != "Durand":
                return False
            
            # Test 5: Context boost
            context = {"last_query_type": "hospitales"}
            decision_with_context = tool.can_handle("y el mapa", context)
            decision_without_context = tool.can_handle("y el mapa", {})
            if decision_with_context.score < decision_without_context.score:
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"Error en test de hospitales: {e}")
            return False
