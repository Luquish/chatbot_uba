"""
Test comprehensivo para contactos_secretarias_tool.py
Prueba la funcionalidad de búsqueda de contactos de hospitales.
"""

import sys
import os
import unittest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tools.contactos_secretarias_tool import ContactosSecretariasTool
from services.sheets_service import SheetsService
from tests.base_test import BaseTest


class TestContactosSecretarias(unittest.TestCase):
    """Test para ContactosSecretariasTool."""
    
    def setUp(self):
        """Configurar test."""
        # No hay setUp en BaseTest, solo inicializar variables
        
        # Mock del SheetsService
        self.sheets_service = None  # Se usará mock en tests reales
        
        # Crear instancia de la tool
        self.tool = ContactosSecretariasTool(self.sheets_service)
        
        # Consultas de test basadas en la estructura de la imagen
        self.test_queries = [
            # Consultas específicas de hospitales
            "¿Cuál es el contacto del hospital Álvarez?",
            "Necesito el mail de Clínicas",
            "¿Dónde puedo contactar al hospital Argerich?",
            "Mail de la primera cátedra de medicina",
            "Contacto de Clínicas 4ta cátedra",
            "Email de Clínicas 6ta cátedra",
            "¿Cómo contacto al hospital Borda?",
            "Mail de Clínicas - Cirugía",
            "Contacto de Clínicas - Dermatología",
            "Email del hospital Churruca",
            
            # Consultas con variaciones
            "contacto clinicas",
            "mail hospital alvarez",
            "email clínicas primera",
            "correo argerich",
            "contacto borda",
            "mail churruca",
            "email clinicas cirugia",
            "contacto clinicas dermatologia",
            
            # Consultas con términos de búsqueda
            "¿Dónde está el contacto de Clínicas?",
            "¿Cuál es el mail de la secretaría de Álvarez?",
            "Necesito el email de contacto del hospital",
            "¿Cómo me contacto con Clínicas?",
            
            # Consultas con números
            "contacto clinicas 1ra",
            "mail clinicas 4ta",
            "email clinicas 6ta",
            "contacto clinicas 7ma",
            
            # Consultas de departamentos
            "mail medicina clinicas",
            "contacto cirugia clinicas",
            "email dermatologia clinicas",
            
            # Consultas que deberían fallar
            "¿Cuál es el horario de clases?",
            "Necesito información sobre cursos",
            "¿Dónde queda la biblioteca?",
            "¿Cuándo es el examen?",
        ]
    
    def test_tool_initialization(self):
        """Test de inicialización de la tool."""
        print("\n🔧 Test de inicialización...")
        
        # Verificar configuración básica
        self.assertEqual(self.tool.name, "sheets.contactos_secretarias")
        self.assertEqual(self.tool.priority, 70)
        
        # Verificar configuración de spreadsheet
        self.assertEqual(
            self.tool.config['spreadsheet_id'], 
            "19t4GasPKs2_Th-2YofT62KgPtfkJDlhLr7fR8Y2iJ4I"
        )
        self.assertEqual(self.tool.config['sheet_name'], 'Hoja 1')
        self.assertEqual(self.tool.config['ranges']['default'], 'A:B')
        
        # Verificar configuración de fuzzy matching
        self.assertTrue(self.tool.config['fuzzy_matching']['enabled'])
        self.assertEqual(self.tool.config['fuzzy_matching']['threshold'], 0.6)
        
        # Verificar configuración de cache
        self.assertTrue(self.tool.config['caching']['enabled'])
        self.assertEqual(self.tool.config['caching']['ttl_minutes'], 60)
        
        print("✅ Inicialización correcta")
    
    def test_rule_score_calculation(self):
        """Test de cálculo de rule score."""
        print("\n📊 Test de rule score...")
        
        # Consultas que deberían tener alto score
        high_score_queries = [
            "contacto hospital alvarez",
            "mail clinicas",
            "email argerich",
            "contacto borda"
        ]
        
        for query in high_score_queries:
            score = self.tool._rule_score(query)
            print(f"  '{query}' -> score: {score:.3f}")
            self.assertGreater(score, 0.3, f"Score muy bajo para: {query}")
        
        # Consultas que deberían tener bajo score
        low_score_queries = [
            "horario de clases",
            "información sobre cursos",
            "biblioteca",
            "examen"
        ]
        
        for query in low_score_queries:
            score = self.tool._rule_score(query)
            print(f"  '{query}' -> score: {score:.3f}")
            self.assertLess(score, 0.3, f"Score muy alto para: {query}")
        
        print("✅ Rule score funcionando correctamente")
    
    def test_query_normalization(self):
        """Test de normalización de consultas."""
        print("\n🔤 Test de normalización...")
        
        test_cases = [
            ("CLINICAS", "clínicas"),
            ("catedra", "cátedra"),
            ("cirugia", "cirugía"),
            ("dermatologia", "dermatología"),
            ("primera", "1ra"),
            ("cuarta", "4ta"),
        ]
        
        for original, expected in test_cases:
            normalized = self.tool._normalize_query(original)
            print(f"  '{original}' -> '{normalized}'")
            # Verificar que se aplicaron los alias
            self.assertIn(expected, normalized)
        
        print("✅ Normalización funcionando correctamente")
    
    def test_hospital_aliases(self):
        """Test de alias de hospitales."""
        print("\n🏥 Test de alias de hospitales...")
        
        # Verificar que los alias están configurados
        expected_aliases = {
            'clinicas': 'clínicas',
            'catedra': 'cátedra',
            'cirugia': 'cirugía',
            'dermatologia': 'dermatología',
            'primera': '1ra',
            'cuarta': '4ta'
        }
        
        for alias, replacement in expected_aliases.items():
            self.assertIn(alias, self.tool.hospital_aliases)
            self.assertEqual(self.tool.hospital_aliases[alias], replacement)
            print(f"  ✅ {alias} -> {replacement}")
        
        print("✅ Alias de hospitales configurados correctamente")
    
    def test_keywords_coverage(self):
        """Test de cobertura de keywords."""
        print("\n🔍 Test de cobertura de keywords...")
        
        # Verificar keywords básicas
        basic_keywords = [
            'contacto', 'mail', 'email', 'correo', 'secretaria', 'secretaría',
            'hospital', 'hospitales', 'udh'
        ]
        
        for keyword in basic_keywords:
            self.assertIn(keyword, self.tool.config['triggers']['keywords'])
            print(f"  ✅ {keyword}")
        
        # Verificar nombres de hospitales
        hospital_names = [
            'aeronautico', 'alvarez', 'alvear', 'argerich', 'avellaneda', 'borda',
            'churruca', 'clinicas', 'clínicas'
        ]
        
        for hospital in hospital_names:
            self.assertIn(hospital, self.tool.config['triggers']['keywords'])
            print(f"  ✅ {hospital}")
        
        # Verificar departamentos
        departments = [
            'medicina', 'cirugia', 'cirugía', 'dermatologia', 'dermatología',
            'primera', 'cuarta', 'quinta', 'sexta', 'septima'
        ]
        
        for dept in departments:
            self.assertIn(dept, self.tool.config['triggers']['keywords'])
            print(f"  ✅ {dept}")
        
        print("✅ Cobertura de keywords completa")
    
    def test_mock_execution(self):
        """Test de ejecución con datos mock."""
        print("\n🎭 Test de ejecución mock...")
        
        # Mock de datos de la hoja (basado en la imagen)
        mock_data = [
            ['HOSPITAL', 'MAIL'],  # Encabezado
            ['AERONAUTICO', 'crodriguezarfelli@fmed.uba.ar'],
            ['ALVAREZ', 'portega@fmed.uba.ar'],
            ['ALVEAR', 'Ibarrio@fmed.uba.ar'],
            ['ARGERICH', 'agaruzzo@fmed.uba.ar'],
            ['AVELLANEDA', 'adelorenzi@fmed.uba.ar'],
            ['BORDA', 'nmuhamed@fmed.uba.ar'],
            ['CHURRUCA', 'mvgallotti@fmed.uba.ar'],
            ['CLINICAS - DPTO MEDICINA', 'magro@fmed.uba.ar'],
            ['CLINICAS 1ra CAT', 'mgduran@fmed.uba.ar'],
            ['CLINICAS 4ta CAT', 'desandor@fmed.uba.ar'],
            ['CLINICAS 5ta CAT', 'svavila@fmed.uba.ar'],
            ['CLINICAS 6ta CAT', 'cmarimon@fmed.uba.ar'],
            ['CLINICAS 7ma CAT', 'sbmartinez@fmed.uba.ar'],
            ['CLINICAS - CIRUGIA', 'catedra.cirugiageneral@fmed.uba.ar'],
            ['CLINICAS - DERMATOLOGIA', 'catedradermatologiaclinicas@gmail.com']
        ]
        
        # Mock del método _fetch_sheet_data
        original_fetch = self.tool._fetch_sheet_data
        self.tool._fetch_sheet_data = lambda: mock_data
        
        try:
            # Test de consultas específicas
            test_cases = [
                ("contacto alvarez", "ALVAREZ", "portega@fmed.uba.ar"),
                ("mail clinicas", "CLINICAS", "magro@fmed.uba.ar"),
                ("email argerich", "ARGERICH", "agaruzzo@fmed.uba.ar"),
                ("contacto borda", "BORDA", "nmuhamed@fmed.uba.ar"),
                ("mail clinicas cirugia", "CLINICAS - CIRUGIA", "catedra.cirugiageneral@fmed.uba.ar"),
                ("email clinicas dermatologia", "CLINICAS - DERMATOLOGIA", "catedradermatologiaclinicas@gmail.com")
            ]
            
            for query, expected_hospital, expected_mail in test_cases:
                response = self.tool.execute(query)
                print(f"\n  Consulta: '{query}'")
                print(f"  Respuesta: {response[:100]}...")
                
                # Verificar que la respuesta contiene información relevante
                self.assertIn("🏥", response, f"No contiene emoji de hospital en: {query}")
                self.assertIn("📧", response, f"No contiene emoji de mail en: {query}")
                
                # Verificar que contiene el hospital esperado (case insensitive)
                self.assertIn(expected_hospital.upper(), response.upper(), 
                             f"No contiene hospital {expected_hospital} en: {query}")
                
                # Verificar que contiene el mail esperado
                self.assertIn(expected_mail, response, 
                             f"No contiene mail {expected_mail} en: {query}")
                
                print(f"  ✅ Encontrado: {expected_hospital} - {expected_mail}")
        
        finally:
            # Restaurar método original
            self.tool._fetch_sheet_data = original_fetch
        
        print("✅ Ejecución mock exitosa")
    
    def test_error_handling(self):
        """Test de manejo de errores."""
        print("\n❌ Test de manejo de errores...")
        
        # Mock que retorna datos vacíos
        self.tool._fetch_sheet_data = lambda: []
        
        response = self.tool.execute("contacto alvarez")
        self.assertIn("❌", response)
        self.assertIn("No se pudieron obtener", response)
        print("  ✅ Manejo de datos vacíos")
        
        # Mock que retorna datos insuficientes
        self.tool._fetch_sheet_data = lambda: [['HOSPITAL']]  # Solo encabezado
        
        response = self.tool.execute("contacto alvarez")
        self.assertIn("❌", response)
        self.assertIn("No hay datos suficientes", response)
        print("  ✅ Manejo de datos insuficientes")
        
        # Mock que simula error de conexión
        def mock_error():
            raise Exception("Error de conexión")
        
        self.tool._fetch_sheet_data = mock_error
        
        response = self.tool.execute("contacto alvarez")
        self.assertIn("❌", response)
        self.assertIn("Error al buscar", response)
        print("  ✅ Manejo de errores de conexión")
        
        print("✅ Manejo de errores funcionando correctamente")
    
    def run_comprehensive_test(self):
        """Ejecutar test comprehensivo."""
        print("🧪 INICIANDO TEST COMPREHENSIVO DE CONTACTOS SECRETARIAS")
        print("=" * 60)
        
        try:
            self.test_tool_initialization()
            self.test_rule_score_calculation()
            self.test_query_normalization()
            self.test_hospital_aliases()
            self.test_keywords_coverage()
            self.test_mock_execution()
            self.test_error_handling()
            
            print("\n" + "=" * 60)
            print("🎉 TEST COMPREHENSIVO COMPLETADO EXITOSAMENTE")
            print("✅ ContactosSecretariasTool está funcionando correctamente")
            
        except Exception as e:
            print(f"\n❌ ERROR EN TEST: {str(e)}")
            raise


if __name__ == "__main__":
    # Ejecutar test
    test = TestContactosSecretarias()
    test.setUp()
    test.run_comprehensive_test()
