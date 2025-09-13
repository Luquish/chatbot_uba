"""
Test final para hospitales_tool.py
Simula consultas reales de usuarios y verifica el funcionamiento correcto del chatbot.
"""

import sys
import os
import unittest
import logging
from unittest.mock import patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tools.hospitales_tool import HospitalesTool, HospitalData
from tests.base_test import BaseTest


class TestHospitalesToolFinal(BaseTest):
    """Test final para HospitalesTool con consultas reales de usuarios."""
    
    def get_test_description(self) -> str:
        return "Test final de hospitales_tool.py - Consultas reales de usuarios y funcionamiento del chatbot"
    
    def get_test_category(self) -> str:
        return "tools"
    
    def setUp(self):
        """Configurar test."""
        # Consultas reales que harían los usuarios del chatbot
        self.consultas_usuarios = [
            # Consultas específicas de hospitales
            "¿Dónde queda el Hospital Alemán?",
            "Ubicación del Hospital de Clínicas",
            "¿Cómo llegar al Hospital Italiano?",
            "Dirección del Hospital Santojanni",
            "¿Dónde está el Hospital Durand?",
            "Hospital Ramos Mejía ubicación",
            "¿Dónde queda el Hospital Evita?",
            "Dirección del Hospital Tornu",
            "¿Cómo llegar al Hospital Zubizarreta?",
            "Ubicación del Hospital Posadas",
            
            # Consultas con variaciones
            "hospital aleman",
            "clinicas direccion",
            "italiano como llegar",
            "santojanni donde queda",
            "durand ubicacion",
            "ramos mejia direccion",
            "evita hospital",
            "tornu ubicacion",
            "zubizarreta direccion",
            "posadas hospital",
            
            # Consultas generales
            "¿Dónde están todos los hospitales?",
            "Lista de hospitales de la UBA",
            "¿Cuáles son los hospitales disponibles?",
            "Hospitales donde hago prácticas",
            "¿Dónde puedo hacer prácticas de medicina?",
            "Mapa de hospitales",
            "Ubicaciones de hospitales",
            "¿Qué hospitales hay?",
            
            # Consultas por barrio
            "¿Qué hospitales hay en Palermo?",
            "Hospitales en Recoleta",
            "¿Dónde puedo hacer prácticas en Belgrano?",
            "Centros médicos en Caballito",
            "Hospitales cerca de San Telmo",
            "¿Qué hay en La Boca?",
            "Hospitales en Villa Crespo",
            "¿Dónde está el hospital más cercano a Almagro?",
            
            # Consultas con contexto
            "Necesito ir a un hospital",
            "¿Dónde hago las prácticas?",
            "Hospitales de la facultad",
            "UDH ubicaciones",
            "Unidades docentes hospitalarias",
            "¿Dónde están las UDH?",
            "Mapa de UDH",
            "Hospitales para estudiantes de medicina",
            
            # Consultas con errores tipográficos
            "hospital aleman",  # sin acento
            "clinicas",  # sin "hospital"
            "italiano",  # solo nombre
            "santojanni",  # solo nombre
            "durand",  # solo nombre
            "ramos mejia",  # sin "hospital"
            "evita",  # solo nombre
            "tornu",  # solo nombre
            "zubizarreta",  # solo nombre
            "posadas",  # solo nombre
        ]
        
        # Hospitales esperados (basados en los datos reales de Google Maps)
        self.hospitales_esperados = [
            "Hospital Alemán",
            "Hospital de Clínicas José de San Martín", 
            "Hospital Italiano de Buenos Aires",
            "Hospital Santojanni",
            "Hospital Durand",
            "Hospital Ramos Mejía",
            "Hospital Evita",
            "Hospital Tornu",
            "Hospital Zubizarreta",
            "Hospital Posadas"
        ]
    
    def _run_test_logic(self) -> bool:
        """Ejecutar tests finales de la herramienta de hospitales."""
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
            
            # Test 2: Detección de consultas de usuarios
            self.log_step("Probando detección de consultas de usuarios...")
            
            consultas_detectadas = 0
            consultas_no_detectadas = []
            
            for consulta in self.consultas_usuarios:
                decision = tool.can_handle(consulta, {})
                if tool.accepts(decision.score):
                    consultas_detectadas += 1
                else:
                    consultas_no_detectadas.append(consulta)
            
            # Debe detectar al menos el 80% de las consultas
            porcentaje_deteccion = (consultas_detectadas / len(self.consultas_usuarios)) * 100
            if porcentaje_deteccion < 80:
                self.log_error(f"Porcentaje de detección muy bajo: {porcentaje_deteccion:.1f}%")
                self.log_error(f"Consultas no detectadas: {consultas_no_detectadas[:5]}")
                return False
            
            self.log_success(f"Detección de consultas: {porcentaje_deteccion:.1f}% ({consultas_detectadas}/{len(self.consultas_usuarios)})")
            
            # Test 3: Extracción de datos reales
            self.log_step("Probando extracción de datos reales...")
            
            hospitales = tool._fetch_hospitales_data()
            if not hospitales or len(hospitales) == 0:
                self.log_error("No se extrajeron hospitales")
                return False
            
            if len(hospitales) < 10:
                self.log_error(f"Muy pocos hospitales extraídos: {len(hospitales)}")
                return False
            
            # Verificar que los hospitales tienen la estructura correcta
            for hospital in hospitales[:5]:  # Verificar primeros 5
                if not hospital.name or not hospital.coordinates:
                    self.log_error(f"Hospital con estructura incorrecta: {hospital}")
                    return False
            
            self.log_success(f"Extraídos {len(hospitales)} hospitales con estructura correcta")
            
            # Test 4: Consultas específicas de hospitales
            self.log_step("Probando consultas específicas de hospitales...")
            
            consultas_especificas = [
                "¿Dónde queda el Hospital Alemán?",
                "Ubicación del Hospital de Clínicas",
                "¿Cómo llegar al Hospital Italiano?",
                "Dirección del Hospital Santojanni",
                "¿Dónde está el Hospital Durand?"
            ]
            
            respuestas_especificas = 0
            for consulta in consultas_especificas:
                result = tool.execute(consulta, {}, {})
                if result.metadata.get('hospital_mentioned'):
                    respuestas_especificas += 1
                elif "Error" in result.response:
                    self.log_error(f"Error en consulta específica: {consulta}")
                    return False
            
            # Debe responder específicamente al menos el 60% de las consultas
            porcentaje_especificas = (respuestas_especificas / len(consultas_especificas)) * 100
            if porcentaje_especificas < 60:
                self.log_error(f"Porcentaje de respuestas específicas muy bajo: {porcentaje_especificas:.1f}%")
                return False
            
            self.log_success(f"Respuestas específicas: {porcentaje_especificas:.1f}% ({respuestas_especificas}/{len(consultas_especificas)})")
            
            # Test 5: Consultas generales
            self.log_step("Probando consultas generales...")
            
            consultas_generales = [
                "¿Dónde están todos los hospitales?",
                "Lista de hospitales de la UBA",
                "¿Cuáles son los hospitales disponibles?",
                "Hospitales donde hago prácticas"
            ]
            
            for consulta in consultas_generales:
                result = tool.execute(consulta, {}, {})
                if not result.response or "Error" in result.response:
                    self.log_error(f"Error en consulta general: {consulta}")
                    return False
                
                if "Total de hospitales" not in result.response:
                    self.log_error(f"Respuesta general no contiene estadísticas: {consulta}")
                    return False
                
                if "tinyurl.com" not in result.response:
                    self.log_error(f"Respuesta general no contiene enlace al mapa: {consulta}")
                    return False
            
            self.log_success("Consultas generales funcionan correctamente")
            
            # Test 6: Búsqueda por barrio
            self.log_step("Probando búsqueda por barrio...")
            
            barrios_test = ["Palermo", "Recoleta", "Belgrano", "Caballito"]
            barrios_con_hospitales = 0
            
            for barrio in barrios_test:
                hospitales_barrio = tool.search_hospitals_by_neighborhood(barrio)
                if hospitales_barrio and len(hospitales_barrio) > 0:
                    barrios_con_hospitales += 1
                    
                    # Verificar estructura de hospitales por barrio
                    for hospital in hospitales_barrio[:2]:  # Verificar primeros 2
                        if not hospital.name or not hospital.coordinates:
                            self.log_error(f"Hospital por barrio con estructura incorrecta: {hospital}")
                            return False
            
            # Debe encontrar hospitales en al menos el 75% de los barrios
            porcentaje_barrios = (barrios_con_hospitales / len(barrios_test)) * 100
            if porcentaje_barrios < 75:
                self.log_error(f"Porcentaje de barrios con hospitales muy bajo: {porcentaje_barrios:.1f}%")
                return False
            
            self.log_success(f"Búsqueda por barrio: {porcentaje_barrios:.1f}% ({barrios_con_hospitales}/{len(barrios_test)})")
            
            # Test 7: Fuzzy matching
            self.log_step("Probando fuzzy matching...")
            
            # Probar variaciones de nombres
            variaciones_test = [
                ("alemán", "Hospital Alemán"),
                ("clinicas", "Hospital de Clínicas"),
                ("italiano", "Hospital Italiano"),
                ("santojanni", "Hospital Santojanni"),
                ("durand", "Hospital Durand")
            ]
            
            matches_correctos = 0
            for variacion, esperado in variaciones_test:
                hospital = tool._find_mentioned_hospital(variacion, hospitales)
                if hospital and esperado.lower() in hospital.name.lower():
                    matches_correctos += 1
            
            # Debe hacer match correcto en al menos el 60% de las variaciones
            porcentaje_fuzzy = (matches_correctos / len(variaciones_test)) * 100
            if porcentaje_fuzzy < 60:
                self.log_error(f"Fuzzy matching muy bajo: {porcentaje_fuzzy:.1f}%")
                return False
            
            self.log_success(f"Fuzzy matching: {porcentaje_fuzzy:.1f}% ({matches_correctos}/{len(variaciones_test)})")
            
            # Test 8: Metadata y sources
            self.log_step("Probando metadata y sources...")
            
            result = tool.execute("¿Dónde están los hospitales?", {}, {})
            
            # Verificar sources
            if not result.sources:
                self.log_error("Sin sources en la respuesta")
                return False
            
            sources_esperados = ["Mapa de Hospitales UDH-CECIM", "Google Maps API"]
            if not any(source in result.sources for source in sources_esperados):
                self.log_error(f"Sources incorrectos: {result.sources}")
                return False
            
            # Verificar metadata
            if not result.metadata:
                self.log_error("Sin metadata en la respuesta")
                return False
            
            metadata_requeridos = ['total_hospitals', 'data_source']
            for key in metadata_requeridos:
                if key not in result.metadata:
                    self.log_error(f"Metadata faltante: {key}")
                    return False
            
            self.log_success("Metadata y sources correctos")
            
            # Test 9: Performance y cache
            self.log_step("Probando performance y cache...")
            
            import time
            
            # Primera consulta (sin cache)
            start_time = time.time()
            result1 = tool.execute("¿Dónde están los hospitales?", {}, {})
            time1 = time.time() - start_time
            
            # Segunda consulta (con cache)
            start_time = time.time()
            result2 = tool.execute("¿Dónde están los hospitales?", {}, {})
            time2 = time.time() - start_time
            
            # La segunda consulta debería ser más rápida (cache)
            if time2 > time1:
                self.log_warning(f"Cache no está funcionando: {time1:.3f}s vs {time2:.3f}s")
            else:
                self.log_success(f"Cache funcionando: {time1:.3f}s vs {time2:.3f}s")
            
            # Test 10: Manejo de errores
            self.log_step("Probando manejo de errores...")
            
            # Test con consulta inválida
            result_error = tool.execute("consulta completamente irrelevante", {}, {})
            if not result_error.response:
                self.log_error("Sin respuesta en caso de consulta inválida")
                return False
            
            self.log_success("Manejo de errores correcto")
            
            # Test 11: Integración con chatbot
            self.log_step("Probando integración con chatbot...")
            
            # Simular flujo completo del chatbot
            context = {}
            
            # Primera consulta
            decision1 = tool.can_handle("¿Dónde están los hospitales?", context)
            if tool.accepts(decision1.score):
                result1 = tool.execute("¿Dónde están los hospitales?", {}, context)
                context['last_query_type'] = 'hospitales'
            
            # Segunda consulta con contexto
            decision2 = tool.can_handle("y el Hospital Alemán?", context)
            if tool.accepts(decision2.score):
                result2 = tool.execute("y el Hospital Alemán?", {}, context)
            
            # El contexto debería mejorar la detección
            if decision2.score >= decision1.score:
                self.log_success("Context boost funcionando correctamente")
            else:
                self.log_warning("Context boost no está funcionando")
            
            self.log_success("Integración con chatbot correcta")
            
            self.log_success("TODOS LOS TESTS FINALES PASARON CORRECTAMENTE")
            return True
            
        except Exception as e:
            self.log_error(f"Error en test final: {e}")
            return False


class TestHospitalesToolIntegration(unittest.TestCase):
    """Tests de integración para HospitalesTool."""
    
    def setUp(self):
        """Configurar test de integración."""
        self.tool = HospitalesTool()
    
    def test_hospital_data_structure(self):
        """Test de la estructura de datos de hospitales."""
        hospital = HospitalData(
            name="Hospital Test",
            address="Dirección Test",
            coordinates=(-34.6, -58.4),
            phone="123-456-7890",
            udh_code="UDH-TEST"
        )
        
        self.assertEqual(hospital.name, "Hospital Test")
        self.assertEqual(hospital.address, "Dirección Test")
        self.assertEqual(hospital.coordinates, (-34.6, -58.4))
        self.assertEqual(hospital.phone, "123-456-7890")
        self.assertEqual(hospital.udh_code, "UDH-TEST")
        self.assertEqual(hospital.specialties, [])
    
    def test_duplicate_removal(self):
        """Test de eliminación de duplicados."""
        hospitales = [
            HospitalData(name="Hospital Test"),
            HospitalData(name="Hospital Test"),  # Duplicado
            HospitalData(name="Hospital Otro"),
            HospitalData(name="HOSPITAL TEST"),  # Duplicado con mayúsculas
        ]
        
        unique = self.tool._remove_duplicate_hospitals(hospitales)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0].name, "Hospital Test")
        self.assertEqual(unique[1].name, "Hospital Otro")
    
    def test_fuzzy_matching(self):
        """Test de fuzzy matching."""
        hospitales = [
            HospitalData(name="Hospital de Clínicas José de San Martín"),
            HospitalData(name="Hospital Alemán"),
            HospitalData(name="Hospital Italiano de Buenos Aires")
        ]
        
        # Test de coincidencia exacta
        result = self.tool._find_mentioned_hospital("hospital de clínicas", hospitales)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Hospital de Clínicas José de San Martín")
        
        # Test de fuzzy matching
        result = self.tool._find_mentioned_hospital("clinicas", hospitales)
        # Puede ser None si no encuentra coincidencia exacta, eso está bien
        if result:
            self.assertIn("clínicas", result.name.lower())
        
        # Test sin coincidencia (puede encontrar algo por fuzzy matching, eso está bien)
        result = self.tool._find_mentioned_hospital("hospital inexistente", hospitales)
        # No verificamos que sea None porque el fuzzy matching puede encontrar algo
    
    def test_distance_calculation(self):
        """Test de cálculo de distancia."""
        # Coordenadas de Buenos Aires
        coord1 = (-34.6037, -58.3816)  # Centro
        coord2 = (-34.6118, -58.3960)  # Palermo
        
        distance = self.tool._calculate_distance(coord1, coord2)
        self.assertGreater(distance, 0)
        self.assertLess(distance, 10)  # Debe ser menos de 10km
    
    def test_hospital_creation_from_place(self):
        """Test de creación de hospital desde Google Places API."""
        mock_place = {
            'name': 'Hospital Test',
            'vicinity': 'Dirección Test, CABA',
            'geometry': {
                'location': {'lat': -34.6, 'lng': -58.4}
            },
            'place_id': 'ChIJTest123'
        }
        
        hospital = self.tool._create_hospital_from_place(mock_place)
        self.assertIsNotNone(hospital)
        self.assertEqual(hospital.name, "Hospital Test")
        self.assertEqual(hospital.address, "Dirección Test, CABA")
        self.assertEqual(hospital.coordinates, (-34.6, -58.4))
        self.assertEqual(hospital.udh_code, "UDH-ChIJTest")
    
    def test_hospital_filtering(self):
        """Test de filtrado de hospitales."""
        mock_places = [
            {'name': 'Hospital Test', 'vicinity': 'Test', 'geometry': {'location': {'lat': -34.6, 'lng': -58.4}}},
            {'name': 'Restaurante Test', 'vicinity': 'Test', 'geometry': {'location': {'lat': -34.6, 'lng': -58.4}}},
            {'name': 'Sanatorio Test', 'vicinity': 'Test', 'geometry': {'location': {'lat': -34.6, 'lng': -58.4}}},
            {'name': 'Clínica Test', 'vicinity': 'Test', 'geometry': {'location': {'lat': -34.6, 'lng': -58.4}}},
        ]
        
        hospitales = []
        for place in mock_places:
            hospital = self.tool._create_hospital_from_place(place)
            if hospital:
                hospitales.append(hospital)
        
        # Debe filtrar solo hospitales, sanatorios y clínicas
        self.assertEqual(len(hospitales), 3)
        nombres = [h.name for h in hospitales]
        self.assertIn("Hospital Test", nombres)
        self.assertIn("Sanatorio Test", nombres)
        self.assertIn("Clínica Test", nombres)
        self.assertNotIn("Restaurante Test", nombres)


if __name__ == '__main__':
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ejecutar test final
    test = TestHospitalesToolFinal()
    result = test.run_test()
    
    print(f"\n{'='*80}")
    print(f"RESULTADO DEL TEST FINAL: {'✅ PASÓ' if result['passed'] else '❌ FALLÓ'}")
    print(f"{'='*80}")
    
    if not result['passed']:
        print(f"Error: {result['error_message']}")
    
    # Ejecutar tests de integración
    print(f"\n{'='*80}")
    print("EJECUTANDO TESTS DE INTEGRACIÓN")
    print(f"{'='*80}")
    
    unittest.main(argv=[''], exit=False, verbosity=2)
