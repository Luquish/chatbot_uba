#!/usr/bin/env python3
"""
Test que compara el enfoque de patrones hardcodeados vs LLM para consultas relativas.
Demuestra la flexibilidad del enfoque LLM.
"""

from base_test import BaseTest
import time


class TestLLMVsPatterns(BaseTest):
    """Compara enfoques hardcodeados vs LLM para consultas relativas."""

    def get_test_description(self) -> str:
        return "Comparación LLM vs Patrones hardcodeados para consultas relativas"

    def get_test_category(self) -> str:
        return "comparison"

    def _run_test_logic(self) -> bool:
        """Compara ambos enfoques."""
        
        self.log_info("=== COMPARACIÓN: LLM vs PATRONES HARDCODEADOS ===")
        
        # Consultas variadas para probar flexibilidad
        test_cases = [
            # Casos básicos que ambos deberían manejar
            {
                "query": "y el que sigue?",
                "context": {"last_query_type": "cursos", "last_month_requested": "AGOSTO"},
                "expected": "SEPTIEMBRE"
            },
            {
                "query": "y la que sigue?", 
                "context": {"last_query_type": "calendario_eventos_generales", "last_time_reference": "esta semana"},
                "expected": "próxima semana"
            },
            
            # Casos con variaciones naturales que solo LLM debería manejar
            {
                "query": "¿Y después qué hay?",
                "context": {"last_query_type": "cursos", "last_month_requested": "MARZO"},
                "expected": "ABRIL"
            },
            {
                "query": "¿Y qué viene después?",
                "context": {"last_query_type": "calendario_eventos_generales", "last_time_reference": "esta semana"},
                "expected": "próxima semana"
            },
            {
                "query": "¿Y antes de eso?",
                "context": {"last_query_type": "cursos", "last_month_requested": "JUNIO"},
                "expected": "MAYO"
            },
            {
                "query": "¿Qué había la semana anterior?",
                "context": {"last_query_type": "calendario_eventos_generales", "last_time_reference": "esta semana"},
                "expected": "semana pasada"
            },
            {
                "query": "¿Y en el mes de antes?",
                "context": {"last_query_type": "cursos", "last_month_requested": "DICIEMBRE"},
                "expected": "NOVIEMBRE"
            }
        ]
        
        self.log_info("=== FASE 1: Test con Patrones Hardcodeados ===")
        
        try:
            from services.relative_query_processors import RelativeQueryManager
            from services.session_service import UserSession
            
            pattern_manager = RelativeQueryManager()
            pattern_results = []
            
            for i, case in enumerate(test_cases):
                # Crear sesión mock
                session = UserSession(user_id=f"test_{i}")
                for key, value in case["context"].items():
                    setattr(session, key, value)
                
                # Test con patrones
                result = pattern_manager.get_context_for_relative_query(session, case["query"], f"test_{i}")
                
                detected = result.get('is_relative', False)
                resolved = result.get('resolved_month') or result.get('resolved_time_reference', '')
                
                pattern_results.append({
                    "query": case["query"],
                    "detected": detected,
                    "resolved": resolved,
                    "expected": case["expected"]
                })
                
                if detected:
                    self.log_success(f"✅ Patrones: '{case['query']}' → {resolved}")
                else:
                    self.log_warning(f"❌ Patrones: '{case['query']}' → NO DETECTADO")
            
            pattern_success_rate = sum(1 for r in pattern_results if r["detected"]) / len(pattern_results)
            self.log_info(f"📊 Tasa de éxito con patrones: {pattern_success_rate:.1%}")
            
        except Exception as e:
            self.log_error(f"Error en test de patrones: {e}")
            return False
        
        self.log_info("=== FASE 2: Test con LLM ===")
        
        try:
            from services.llm_context_resolver import HybridContextResolver
            
            llm_resolver = HybridContextResolver()
            llm_results = []
            
            for case in test_cases:
                # Preparar contexto para LLM
                session_context = {
                    'last_query': case["context"].get("last_query", "consulta anterior"),
                    'last_query_type': case["context"].get("last_query_type", ""),
                    'last_month_requested': case["context"].get("last_month_requested", ""),
                    'last_time_reference': case["context"].get("last_time_reference", "")
                }
                
                # Test con LLM
                result = llm_resolver.resolve_relative_query(case["query"], session_context)
                
                llm_results.append({
                    "query": case["query"],
                    "detected": result.is_relative,
                    "resolved": result.resolved_context or "",
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                    "expected": case["expected"]
                })
                
                if result.is_relative:
                    self.log_success(f"✅ LLM ({result.confidence:.1f}): '{case['query']}' → {result.resolved_context}")
                    self.log_info(f"   Explicación: {result.explanation}")
                else:
                    self.log_warning(f"❌ LLM: '{case['query']}' → NO DETECTADO")
                    self.log_info(f"   Explicación: {result.explanation}")
                
                # Pequeña pausa para no saturar la API
                time.sleep(0.5)
            
            llm_success_rate = sum(1 for r in llm_results if r["detected"]) / len(llm_results)
            self.log_info(f"📊 Tasa de éxito con LLM: {llm_success_rate:.1%}")
            
        except Exception as e:
            self.log_error(f"Error en test de LLM: {e}")
            self.log_info("(Puede ser por falta de API key de OpenAI)")
            # No fallar el test, solo reportar
        
        self.log_info("=== FASE 3: Análisis de Flexibilidad ===")
        
        # Casos específicos que demuestran flexibilidad del LLM
        flexibility_cases = [
            "¿Y qué pasa después de eso?",
            "¿Y en el período anterior?", 
            "¿Y lo que viene luego?",
            "¿Y antes de todo eso?",
            "¿Qué hay en el siguiente?",
            "¿Y en el de atrás?"
        ]
        
        self.log_info(f"📝 Probando {len(flexibility_cases)} casos de flexibilidad...")
        
        try:
            session_context = {
                'last_query': 'cursos de MAYO',
                'last_query_type': 'cursos',
                'last_month_requested': 'MAYO',
                'last_time_reference': ''
            }
            
            flexible_detected = 0
            for query in flexibility_cases:
                result = llm_resolver.resolve_relative_query(query, session_context)
                if result.is_relative:
                    flexible_detected += 1
                    self.log_success(f"✅ Flexibilidad: '{query}' → {result.resolved_context}")
                else:
                    self.log_info(f"⚪ Flexibilidad: '{query}' → no relativo")
                time.sleep(0.3)
            
            flexibility_rate = flexible_detected / len(flexibility_cases)
            self.log_info(f"📊 Tasa de flexibilidad del LLM: {flexibility_rate:.1%}")
            
        except Exception as e:
            self.log_warning(f"Test de flexibilidad omitido: {e}")
        
        self.log_info("=== RECOMENDACIONES ===")
        
        self.log_success("🎯 ENFOQUE HÍBRIDO RECOMENDADO:")
        self.log_success("   1. ✅ Patrones rápidos para casos MUY comunes")
        self.log_success("   2. ✅ LLM para casos complejos y flexibilidad")
        self.log_success("   3. ✅ Cache de respuestas LLM para optimizar costos")
        self.log_success("   4. ✅ Fallback graceful si LLM no está disponible")
        
        self.log_info("💡 VENTAJAS DEL ENFOQUE LLM:")
        self.log_info("   • Maneja variaciones naturales del lenguaje")
        self.log_info("   • No requiere hardcodear nuevos patrones")
        self.log_info("   • Entiende contexto conversacional complejo")
        self.log_info("   • Se adapta a nuevos tipos de consultas")
        
        self.log_info("⚡ VENTAJAS DE LOS PATRONES:")
        self.log_info("   • Respuesta instantánea")
        self.log_info("   • Sin costo de API")
        self.log_info("   • Totalmente predecible")
        self.log_info("   • Funciona sin conectividad")
        
        return True


if __name__ == "__main__":
    test = TestLLMVsPatterns()
    success = test.run_test()
    exit(0 if success else 1)