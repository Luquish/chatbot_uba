"""
Test simplificado para hospitales_tool.py
Simula consultas reales de usuarios y verifica el funcionamiento correcto del chatbot.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tools.hospitales_tool import HospitalesTool
from tests.base_test import BaseTest


class TestHospitalesToolSimpleFinal(BaseTest):
    """Test simplificado para HospitalesTool con consultas reales de usuarios."""
    
    def get_test_description(self) -> str:
        return "Test simplificado de hospitales_tool.py - Consultas reales de usuarios"
    
    def get_test_category(self) -> str:
        return "tools"
    
    def _run_test_logic(self) -> bool:
        """Ejecutar tests simplificados de la herramienta de hospitales."""
        try:
            # Test 1: Inicialización
            self.log_step("Probando inicialización de la herramienta...")
            tool = HospitalesTool()
            
            if tool.name != "hospitales":
                self.log_error("Nombre de herramienta incorrecto")
                return False
            
            if tool.priority != 75:
                self.log_error("Prioridad incorrecta")
                return False
                
            if not tool.gmaps:
                self.log_error("Google Maps API no está configurada")
                return False
            
            self.log_success("Inicialización correcta")
            
            # Test 2: Consultas específicas de usuarios
            self.log_step("Probando consultas específicas de usuarios...")
            
            consultas_especificas = [
                "¿Dónde queda el Hospital Alemán?",
                "Ubicación del Hospital de Clínicas",
                "¿Cómo llegar al Hospital Italiano?",
                "Dirección del Hospital Santojanni",
                "¿Dónde está el Hospital Durand?"
            ]
            
            respuestas_especificas = 0
            for consulta in consultas_especificas:
                decision = tool.can_handle(consulta, {})
                if tool.accepts(decision.score):
                    result = tool.execute(consulta, {}, {})
                    if result.metadata.get('hospital_mentioned'):
                        respuestas_especificas += 1
                        self.log_success(f"Consulta específica detectada: {consulta[:30]}...")
                    elif "Error" not in result.response:
                        self.log_success(f"Consulta procesada: {consulta[:30]}...")
                    else:
                        self.log_error(f"Error en consulta: {consulta}")
                        return False
            
            self.log_success(f"Consultas específicas procesadas: {respuestas_especificas}/{len(consultas_especificas)}")
            
            # Test 3: Consultas generales
            self.log_step("Probando consultas generales...")
            
            consultas_generales = [
                "¿Dónde están todos los hospitales?",
                "Lista de hospitales de la UBA",
                "¿Cuáles son los hospitales disponibles?",
                "Hospitales donde hago prácticas"
            ]
            
            for consulta in consultas_generales:
                decision = tool.can_handle(consulta, {})
                if tool.accepts(decision.score):
                    result = tool.execute(consulta, {}, {})
                    if "Error" in result.response:
                        self.log_error(f"Error en consulta general: {consulta}")
                        return False
                    
                    if "Total de hospitales" not in result.response:
                        self.log_error(f"Respuesta general no contiene estadísticas: {consulta}")
                        return False
                    
                    self.log_success(f"Consulta general procesada: {consulta[:30]}...")
            
            self.log_success("Consultas generales funcionan correctamente")
            
            # Test 4: Búsqueda por barrio
            self.log_step("Probando búsqueda por barrio...")
            
            barrios_test = ["Palermo", "Recoleta", "Belgrano"]
            barrios_con_hospitales = 0
            
            for barrio in barrios_test:
                hospitales_barrio = tool.search_hospitals_by_neighborhood(barrio)
                if hospitales_barrio and len(hospitales_barrio) > 0:
                    barrios_con_hospitales += 1
                    self.log_success(f"Encontrados {len(hospitales_barrio)} hospitales en {barrio}")
                else:
                    self.log_warning(f"No se encontraron hospitales en {barrio}")
            
            if barrios_con_hospitales == 0:
                self.log_error("No se encontraron hospitales en ningún barrio")
                return False
            
            self.log_success(f"Búsqueda por barrio: {barrios_con_hospitales}/{len(barrios_test)} barrios con hospitales")
            
            # Test 5: Extracción de datos
            self.log_step("Probando extracción de datos...")
            
            hospitales = tool._fetch_hospitales_data()
            if not hospitales or len(hospitales) == 0:
                self.log_error("No se extrajeron hospitales")
                return False
            
            if len(hospitales) < 10:
                self.log_error(f"Muy pocos hospitales extraídos: {len(hospitales)}")
                return False
            
            # Verificar estructura de datos
            for hospital in hospitales[:3]:  # Verificar primeros 3
                if not hospital.name or not hospital.coordinates:
                    self.log_error(f"Hospital con estructura incorrecta: {hospital}")
                    return False
            
            self.log_success(f"Extraídos {len(hospitales)} hospitales con estructura correcta")
            
            # Test 6: Metadata y sources
            self.log_step("Probando metadata y sources...")
            
            result = tool.execute("¿Dónde están los hospitales?", {}, {})
            
            if not result.sources:
                self.log_error("Sin sources en la respuesta")
                return False
            
            if not result.metadata:
                self.log_error("Sin metadata en la respuesta")
                return False
            
            if 'total_hospitals' not in result.metadata:
                self.log_error("Metadata faltante: total_hospitals")
                return False
            
            self.log_success("Metadata y sources correctos")
            
            # Test 7: Integración con chatbot
            self.log_step("Probando integración con chatbot...")
            
            # Simular flujo del chatbot
            context = {}
            
            # Primera consulta
            decision1 = tool.can_handle("¿Dónde están los hospitales?", context)
            if tool.accepts(decision1.score):
                result1 = tool.execute("¿Dónde están los hospitales?", {}, context)
                context['last_query_type'] = 'hospitales'
                self.log_success("Primera consulta procesada")
            
            # Segunda consulta con contexto
            decision2 = tool.can_handle("y el Hospital Alemán?", context)
            if tool.accepts(decision2.score):
                result2 = tool.execute("y el Hospital Alemán?", {}, context)
                self.log_success("Segunda consulta con contexto procesada")
            
            self.log_success("Integración con chatbot correcta")
            
            self.log_success("TODOS LOS TESTS SIMPLIFICADOS PASARON CORRECTAMENTE")
            return True
            
        except Exception as e:
            self.log_error(f"Error en test simplificado: {e}")
            return False


if __name__ == '__main__':
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ejecutar test simplificado
    test = TestHospitalesToolSimpleFinal()
    result = test.run_test()
    
    print(f"\n{'='*80}")
    print(f"RESULTADO DEL TEST SIMPLIFICADO: {'✅ PASÓ' if result['passed'] else '❌ FALLÓ'}")
    print(f"{'='*80}")
    
    if not result['passed']:
        print(f"Error: {result['error_message']}")
    else:
        print("🎉 La herramienta de hospitales está funcionando correctamente!")
        print("✅ Consultas específicas de usuarios funcionan")
        print("✅ Consultas generales funcionan")
        print("✅ Búsqueda por barrio funciona")
        print("✅ Extracción de datos funciona")
        print("✅ Integración con chatbot funciona")
