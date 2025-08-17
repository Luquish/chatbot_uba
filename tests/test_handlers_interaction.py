#!/usr/bin/env python3
"""
Test de interacción completa con todos los handlers del sistema.
Valida FAQs, Calendar, Sheets, Intents y Telegram handlers.
"""

import os
import sys
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, date
from typing import Dict, Any, List

from base_test import BaseTest

# Importar handlers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from handlers.faqs_handler import handle_faq_query, get_faq_intent
from handlers.calendar_handler import get_calendar_events
from handlers.courses_handler import handle_sheet_course_query, parse_sheet_course_data
from handlers.intent_handler import get_query_intent, handle_conversational_intent
from handlers.telegram_handler import TelegramHandler

# Importar servicios para mocks
from services.calendar_service import CalendarService
from services.sheets_service import SheetsService
from utils.date_utils import DateUtils


class TestHandlersInteraction(BaseTest):
    """Test de interacción completa con todos los handlers."""
    
    def get_test_description(self) -> str:
        return "Test de interacción con todos los handlers (FAQs, Calendar, Sheets, Intents, Telegram)"
    
    def get_test_category(self) -> str:
        return "handlers"
    
    def _run_test_logic(self) -> bool:
        """Validar todos los handlers del sistema."""
        print("🛠️ Probando interacción con todos los handlers...")
        
        try:
            # Ejecutar tests de cada handler
            results = []
            
            results.append(self._test_faqs_handler())
            results.append(self._test_calendar_handler())
            results.append(self._test_courses_handler())
            results.append(self._test_intent_handler())
            results.append(self._test_telegram_handler())
            
            successful_tests = sum(results)
            total_tests = len(results)
            
            print(f"\n✅ Tests de handlers completados: {successful_tests}/{total_tests} exitosos")
            
            # Considerar exitoso si al menos 4 de 5 handlers funcionan
            return successful_tests >= 4
            
        except Exception as e:
            self.log_error(f"Error en test de handlers: {str(e)}")
            return False
    
    def _test_faqs_handler(self) -> bool:
        """Test del handler de FAQs."""
        print("\n📋 Probando FAQs Handler...")
        
        try:
            # Casos de prueba para FAQs
            faq_test_cases = [
                {
                    'query': '¿Cómo obtengo la constancia de alumno regular?',
                    'should_contain': ['constancia', 'regular', 'ventanilla', 'inscripciones']
                },
                {
                    'query': '¿Cómo doy de baja una materia?',
                    'should_contain': ['baja', 'materia', 'parcial', 'inscripciones']
                },
                {
                    'query': '¿Cómo anulo la inscripción a un final?',
                    'should_contain': ['anular', 'final', 'ventanilla', 'constancia']
                },
                {
                    'query': '¿Cómo me reincorporo a la carrera?',
                    'should_contain': ['reincorporación', 'carrera', 'inscripciones', 'automático']
                },
                {
                    'query': '¿Cómo me inscribo para recursar?',
                    'should_contain': ['recursada', 'trámite', 'inscripciones', 'baja']
                }
            ]
            
            successful_cases = 0
            
            for i, case in enumerate(faq_test_cases, 1):
                print(f"  📝 FAQ Caso {i}: {case['query']}")
                
                # Test get_faq_intent
                faq_intent = get_faq_intent(case['query'])
                if faq_intent and faq_intent['confidence'] > 0.25:
                    print(f"     - Intent detectado: {faq_intent['intent']} (confianza: {faq_intent['confidence']:.2f})")
                    
                    # Test handle_faq_query
                    response = handle_faq_query(case['query'])
                    if response:
                        response_lower = response.lower()
                        contains_expected = any(keyword.lower() in response_lower for keyword in case['should_contain'])
                        
                        print(f"     📝 Respuesta FAQ: {response[:200]}...")
                        
                        if contains_expected:
                            print(f"     ✅ Respuesta FAQ válida")
                            successful_cases += 1
                        else:
                            print(f"     ⚠️ Respuesta no contiene elementos esperados: {case['should_contain']}")
                    else:
                        print(f"     ⚠️ No se generó respuesta FAQ")
                else:
                    print(f"     ⚠️ No se detectó intent FAQ válido")
            
            print(f"  📋 FAQs Handler: {successful_cases}/{len(faq_test_cases)} casos exitosos")
            return successful_cases >= len(faq_test_cases) * 0.6  # 60% éxito mínimo
            
        except Exception as e:
            print(f"  ❌ Error en FAQs Handler: {str(e)}")
            return False
    
    def _test_calendar_handler(self) -> bool:
        """Test del handler de calendario."""
        print("\n📅 Probando Calendar Handler...")
        
        try:
            # Mock del CalendarService
            mock_calendar_service = Mock(spec=CalendarService)
            
            # Mock de eventos de prueba
            mock_events = [
                {
                    'summary': 'Examen Final Anatomía',
                    'start': '15/12/2024 09:00',
                    'end': '15/12/2024 12:00',
                    'same_day': True,
                    'calendar_type': 'examenes',
                    'description': 'Examen final de anatomía'
                },
                {
                    'summary': 'Inscripción a Finales',
                    'start': '01/11/2024',
                    'end': '10/11/2024',
                    'same_day': False,
                    'calendar_type': 'inscripciones'
                }
            ]
            
            # Configurar mocks
            mock_calendar_service.get_events_this_week.return_value = mock_events
            mock_calendar_service.get_events_by_type.return_value = mock_events
            mock_calendar_service.get_upcoming_events.return_value = mock_events
            
            # Test casos
            test_cases = [
                {'intent': None, 'expected_events': 2},
                {'intent': 'examenes', 'expected_events': 2},
                {'intent': 'inscripciones', 'expected_events': 2}
            ]
            
            successful_cases = 0
            
            for i, case in enumerate(test_cases, 1):
                print(f"  📝 Calendar Caso {i}: intent={case['intent']}")
                
                response = get_calendar_events(mock_calendar_service, case['intent'])
                
                if response and 'Eventos encontrados' in response:
                    # Contar eventos en la respuesta (buscar emojis de eventos)
                    event_count = response.count('📝') + response.count('✍️') + response.count('📚') + response.count('📋') + response.count('📌')
                    
                    if event_count >= case['expected_events']:
                        print(f"     ✅ Respuesta con {event_count} eventos")
                        successful_cases += 1
                    else:
                        print(f"     ⚠️ Respuesta con solo {event_count} eventos")
                else:
                    print(f"     ⚠️ Respuesta no válida o vacía")
            
            print(f"  📅 Calendar Handler: {successful_cases}/{len(test_cases)} casos exitosos")
            return successful_cases >= len(test_cases) * 0.6
            
        except Exception as e:
            print(f"  ❌ Error en Calendar Handler: {str(e)}")
            return False
    
    def _test_courses_handler(self) -> bool:
        """Test del handler de cursos/sheets."""
        print("\n📊 Probando Courses/Sheets Handler...")
        
        try:
            # Mock del SheetsService
            mock_sheets_service = Mock(spec=SheetsService)
            
            # Mock de datos de sheet de prueba
            mock_sheet_data = [
                ['NOMBRE DE ACTIVIDAD', 'FORMULARIO', 'FECHA', 'HORARIO', 'MODALIDAD'],
                ['Curso de Primeros Auxilios', 'https://forms.google.com/primeros-auxilios', '15/12', '09:00-12:00', 'Presencial'],
                ['Curso de RCP', 'https://forms.google.com/rcp', '20/12', '14:00-17:00', 'Presencial'],
                ['Curso de Suturas', '', '25/12', '10:00-13:00', 'Presencial']
            ]
            
            mock_sheets_service.get_sheet_values.return_value = mock_sheet_data
            
            # Mock DateUtils
            mock_date_utils = Mock(spec=DateUtils)
            mock_date_utils.get_today.return_value = date(2024, 12, 10)
            mock_date_utils.get_current_weekday_name.return_value = 'martes'
            mock_date_utils.parse_date_from_text.return_value = None
            mock_date_utils.get_courses_for_this_week.return_value = []
            mock_date_utils.get_courses_for_weekday.return_value = []
            
            # Mock para sort_courses_by_proximity - simular ordenamiento
            def mock_sort_courses(courses, date_utils):
                return sorted(courses, key=lambda x: x.get('nombre_actividad', ''))
            
            # Patch la función sort_courses_by_proximity
            with patch('handlers.courses_handler.sort_courses_by_proximity', side_effect=mock_sort_courses):
                
                # Test parse_sheet_course_data
                print("  📝 Probando parsing de datos de sheet...")
                parsed_courses = parse_sheet_course_data(mock_sheet_data)
                
                if len(parsed_courses) >= 2:  # Esperamos al menos 2 cursos
                    print(f"     ✅ Parseados {len(parsed_courses)} cursos correctamente")
                    parsing_success = True
                else:
                    print(f"     ⚠️ Solo se parsearon {len(parsed_courses)} cursos")
                    parsing_success = False
                
                # Test handle_sheet_course_query
                print("  📝 Probando consultas de cursos...")
                
                course_queries = [
                    '¿Hay cursos de primeros auxilios?',
                    '¿Dónde está el formulario del curso de RCP?',
                    '¿Qué cursos hay disponibles?',
                    'formulario del curso de suturas'
                ]
                
                successful_queries = 0
                
                for query in course_queries:
                    response = handle_sheet_course_query(
                        query, 
                        mock_sheets_service, 
                        'test_spreadsheet_id', 
                        mock_date_utils
                    )
                    
                    if response and len(response) > 50:  # Respuesta sustancial
                        print(f"     ✅ Consulta procesada: {query[:30]}...")
                        successful_queries += 1
                    else:
                        print(f"     ⚠️ Consulta no procesada: {query[:30]}...")
                
                # Resultado combinado
                total_success = parsing_success and (successful_queries >= len(course_queries) * 0.5)
                
                print(f"  📊 Courses Handler: parsing={'✅' if parsing_success else '❌'}, queries={successful_queries}/{len(course_queries)}")
                return total_success
            
        except Exception as e:
            print(f"  ❌ Error en Courses Handler: {str(e)}")
            return False
    
    def _test_intent_handler(self) -> bool:
        """Test del handler de intenciones."""
        print("\n🧠 Probando Intent Handler...")
        
        try:
            # Casos de prueba para detección de intenciones
            intent_test_cases = [
                {'query': 'hola', 'expected_intent': 'saludo', 'min_confidence': 0.7},
                {'query': '¿cómo estás?', 'expected_intent': 'cortesia', 'min_confidence': 0.7},
                {'query': 'gracias', 'expected_intent': 'agradecimiento', 'min_confidence': 0.7},
                {'query': '¿quién eres?', 'expected_intent': 'identidad', 'min_confidence': 0.7},
                {'query': '¿qué puedes hacer?', 'expected_intent': 'pregunta_capacidades', 'min_confidence': 0.5},
                {'query': 'me duele la cabeza', 'expected_intent': 'consulta_medica', 'min_confidence': 0.5}
            ]
            
            successful_intents = 0
            
            for i, case in enumerate(intent_test_cases, 1):
                print(f"  📝 Intent Caso {i}: {case['query']}")
                
                # Test get_query_intent (sin embedding model para simplicidad)
                intent, confidence = get_query_intent(case['query'], embedding_model=None)
                
                print(f"     - Intent detectado: {intent} (confianza: {confidence:.2f})")
                
                if intent == case['expected_intent'] and confidence >= case['min_confidence']:
                    print(f"     ✅ Intent correcto")
                    successful_intents += 1
                else:
                    print(f"     ⚠️ Intent incorrecto o baja confianza")
                
                # Test handle_conversational_intent
                response_dict = handle_conversational_intent(intent, confidence, case['query'])
                
                if response_dict and response_dict.get('response'):
                    print(f"     ✅ Respuesta conversacional generada")
                else:
                    print(f"     ℹ️ No se generó respuesta conversacional (normal para algunos casos)")
            
            print(f"  🧠 Intent Handler: {successful_intents}/{len(intent_test_cases)} intents correctos")
            return successful_intents >= len(intent_test_cases) * 0.6
            
        except Exception as e:
            print(f"  ❌ Error en Intent Handler: {str(e)}")
            return False
    
    def _test_telegram_handler(self) -> bool:
        """Test del handler de Telegram."""
        print("\n🤖 Probando Telegram Handler...")
        
        try:
            # Crear instancia de TelegramHandler con datos de prueba
            telegram_handler = TelegramHandler(
                bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"  # Token de prueba válido en formato
            )
            
            # Test de configuración básica
            print("  📝 Probando configuración básica...")
            
            # Verificar que los atributos esenciales están configurados
            attributes_check = all([
                hasattr(telegram_handler, 'bot_token'),
                hasattr(telegram_handler, 'api_url'),
                hasattr(telegram_handler, 'webhook_secret'),
                telegram_handler.api_url.startswith('https://api.telegram.org/bot')
            ])
            
            if attributes_check:
                print(f"     ✅ Telegram handler configurado correctamente")
                structure_success = True
            else:
                print(f"     ⚠️ Telegram handler mal configurado")
                structure_success = False
            
            # Test de parseo de mensaje (simulado)
            print("  📝 Probando parseo de mensaje...")
            
            # Simular datos de mensaje de Telegram
            mock_message_data = {
                "message": {
                    "message_id": 123,
                    "from": {
                        "id": 987654321,
                        "first_name": "Juan",
                        "last_name": "Pérez",
                        "username": "juanperez"
                    },
                    "chat": {
                        "id": 987654321,
                        "type": "private"
                    },
                    "date": 1640995200,
                    "text": "Hola, ¿cómo están?"
                }
            }
            
            # Simular el parseo (sin hacer request real)
            try:
                # Solo verificamos que los métodos existen
                methods_exist = all([
                    hasattr(telegram_handler, 'parse_message'),
                    hasattr(telegram_handler, 'send_message'),
                    hasattr(telegram_handler, 'validate_webhook'),
                    hasattr(telegram_handler, 'get_me')
                ])
                
                if methods_exist:
                    print(f"     ✅ Métodos esenciales disponibles")
                    methods_success = True
                else:
                    print(f"     ⚠️ Faltan métodos esenciales")
                    methods_success = False
                    
            except Exception as e:
                print(f"     ⚠️ Error en test de métodos: {str(e)}")
                methods_success = False
            
            # Test de URL API
            print("  📝 Probando URL de API...")
            expected_url = "https://api.telegram.org/bot123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
            url_correct = telegram_handler.api_url == expected_url
            
            if url_correct:
                print(f"     ✅ URL de API configurada correctamente")
            else:
                print(f"     ⚠️ URL incorrecta. Esperada: {expected_url}, Actual: {telegram_handler.api_url}")
            
            # Resultado combinado
            total_success = structure_success and methods_success and url_correct
            
            print(f"  🤖 Telegram Handler: estructura={'✅' if structure_success else '❌'}, métodos={'✅' if methods_success else '❌'}, URL={'✅' if url_correct else '❌'}")
            return total_success
            
        except Exception as e:
            print(f"  ❌ Error en Telegram Handler: {str(e)}")
            return False


if __name__ == "__main__":
    test = TestHandlersInteraction()
    result = test.run_test()
    print(f"\nResultado final: {'✅ ÉXITO' if result['passed'] else '❌ FALLO'}")
    if not result['passed']:
        print(f"Error: {result['error_message']}")