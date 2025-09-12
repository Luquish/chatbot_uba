#!/usr/bin/env python3
"""
Test comprensivo para evaluar el funcionamiento de la herramienta de secretarías.
Evalúa consultas específicas y generales sobre secretarías de cátedras y departamentos.
"""

import time
from base_test import BaseTest


class TestSecretariasComprehensive(BaseTest):
    """Test comprensivo para la herramienta de secretarías."""

    def get_test_description(self) -> str:
        return "Test comprensivo de secretarías - consultas específicas y generales"

    def get_test_category(self) -> str:
        return "secretarias"

    def _run_test_logic(self) -> bool:
        """Ejecuta el test comprensivo de secretarías."""
        
        self.log_info("=== TEST COMPRENSIVO DE SECRETARÍAS ===")
        
        # Lista completa de consultas a probar
        queries = [
            # Cátedras específicas
            "¿Dónde está la secretaría de Anatomia Cátedra 1?",
            "¿Dónde está la secretaría de la Cátedra 2?",
            "¿Dónde está la secretaría de la Cátedra 3?",
            "¿Dónde está la secretaría de Histologia Y Embriologia (Hye) Cátedra HISTOLOGIA?",
            "¿Dónde está la secretaría de la Cátedra EMBRIOLOGIO?",
            "¿Dónde está la secretaría de la Cátedra BIO Y GEN?",
            "¿Dónde está la secretaría de Bioetica 1 Y 2 Cátedra?",
            "¿Dónde está la secretaría de Salud Mental Cátedra?",
            "¿Dónde está la secretaría de Fisio Cátedra 1?",
            "¿Dónde está la secretaría de la Cátedra 2?",
            "¿Dónde está la secretaría de Bioquimica Cátedra 1?",
            "¿Dónde está la secretaría de la Cátedra 2?",
            "¿Dónde está la secretaría de Microbiologia 1 Y 2 Cátedra 1?",
            "¿Dónde está la secretaría de la Cátedra 2?",
            "¿Dónde está la secretaría de Farmacologia 1 Y 2 Cátedra 1?",
            "¿Dónde está la secretaría de la Cátedra 2?",
            "¿Dónde está la secretaría de la Cátedra 3?",
            "¿Dónde está la secretaría de Toxicologia Cátedra 1?",
            "¿Dónde está la secretaría de Medicina Legal Cátedra?",
            "¿Dónde está la secretaría de Salud Publica 1 Y 2 Cátedra?",
            "¿Dónde está la secretaría de Patologia 1 Y 2 Cátedra?",
            "¿Dónde está la secretaría de Familiar?",
            "¿Dónde está la secretaría de Inmunologia Cátedra 1?",
            "¿Dónde está la secretaría de la Cátedra 2?",
            
            # Secretarías administrativas
            "¿Dónde queda la secretaría de Alumnos?",
            "¿Dónde queda la secretaría de Admisión?",
            "¿Dónde queda la secretaría de Conexas?",
            "¿Dónde queda la secretaría de Graduados?",
            "¿Dónde queda la secretaría de Posgrados?",
            "¿Dónde queda la secretaría de Títulos?",
            "¿Dónde queda la secretaría de Tesorería?",
            
            # Secretarías de carreras
            "¿Dónde queda la secretaría de Sec. Enfermería?",
            "¿Dónde queda la secretaría de Sec. Obstetricia?",
            "¿Dónde queda la secretaría de Sec. Hemoterapia?",
            "¿Dónde queda la secretaría de Sec. Inst. Quir.?",
            "¿Dónde queda la secretaría de Sec. Podología?",
            "¿Dónde queda la secretaría de Sec. Kinesiología?",
            "¿Dónde queda la secretaría de Sec. Nutrición?",
            "¿Dónde queda la secretaría de Sec. Cosmetología?",
            "¿Dónde queda la secretaría de Sec. Prac. Cardio.?",
            "¿Dónde queda la secretaría de Sec. Fonoaudiología?",
            "¿Dónde queda la secretaría de Sec. Radiología?",
            
            # Departamentos y servicios
            "¿Dónde queda la secretaría de Museo Naón?",
            "¿Dónde queda la secretaría de UBA XXI?",
            "¿Dónde queda la secretaría de Depto. Discapacidad?",
            "¿Dónde queda la secretaría de Depto. Género?",
            "¿Dónde queda la secretaría de SEUBE?",
            "¿Dónde queda la secretaría de PASES Y SIMULTANEIDAD?",
            "¿Dónde queda la secretaría de CICLO BIOMEDICO?",
            "¿Dónde queda la secretaría de CICLO CLINICO?",
            "¿Dónde queda la secretaría de ARCHIVOS?",
            "¿Dónde queda la secretaría de ACTAS Y CERTIFICACIONES?"
        ]
        
        # Inicializar el sistema RAG
        try:
            from rag_system import RAGSystem
            rag = RAGSystem()
            self.log_success("✅ Sistema RAG inicializado correctamente")
        except Exception as e:
            self.log_error(f"❌ Error inicializando RAG: {e}")
            return False
        
        # Contadores para estadísticas
        total_queries = len(queries)
        successful_queries = 0
        tool_usage = {}
        response_lengths = []
        
        self.log_info(f"📊 Ejecutando {total_queries} consultas de secretarías...")
        
        # Ejecutar cada consulta
        for i, query in enumerate(queries, 1):
            self.log_info(f"\\n📋 CONSULTA {i}/{total_queries}: \"{query}\"")
            
            try:
                # Procesar consulta
                start_time = time.time()
                result = rag.process_query(query, user_id=f'test_secretarias_{i}', user_name='Test User')
                end_time = time.time()
                
                # Extraer información del resultado
                query_type = result.get('query_type', 'unknown')
                response = result.get('response', '')
                response_length = len(response)
                
                # Contar uso de herramientas
                if query_type in tool_usage:
                    tool_usage[query_type] += 1
                else:
                    tool_usage[query_type] = 1
                
                # Verificar si la consulta fue exitosa (criterio mejorado)
                is_successful = (
                    query_type == 'router_sheets.horarios_secretarias' and 
                    response_length > 30 and
                    ('HORARIOS DE SECRETARÍAS' in response or 
                     '📍' in response or  # Respuestas específicas con emoji de ubicación
                     'SECTOR' in response or  # Contiene información de sector
                     'PISO' in response or  # Contiene información de piso
                     'MAIL' in response)  # Contiene información de contacto
                )
                
                if is_successful:
                    successful_queries += 1
                    self.log_success(f"✅ {query_type} - {response_length} chars - {end_time-start_time:.2f}s")
                else:
                    self.log_warning(f"⚠️ {query_type} - {response_length} chars - {end_time-start_time:.2f}s")
                
                response_lengths.append(response_length)
                
                # Mostrar preview de respuesta para consultas problemáticas
                if not is_successful and response_length > 0:
                    preview = response[:150].replace('\\n', ' ')
                    self.log_info(f"   Preview: {preview}...")
                
            except Exception as e:
                self.log_error(f"❌ Error en consulta {i}: {e}")
                response_lengths.append(0)
        
        # Generar reporte final
        self.log_info("\\n" + "="*80)
        self.log_info("📊 REPORTE FINAL DEL TEST DE SECRETARÍAS")
        self.log_info("="*80)
        
        # Estadísticas generales
        success_rate = (successful_queries / total_queries) * 100
        avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        self.log_info(f"📈 Estadísticas Generales:")
        self.log_info(f"   Total consultas: {total_queries}")
        self.log_info(f"   Consultas exitosas: {successful_queries}")
        self.log_info(f"   Tasa de éxito: {success_rate:.1f}%")
        self.log_info(f"   Longitud promedio de respuesta: {avg_response_length:.0f} caracteres")
        
        # Uso de herramientas
        self.log_info(f"\\n🔧 Uso de Herramientas:")
        for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_queries) * 100
            self.log_info(f"   {tool}: {count} consultas ({percentage:.1f}%)")
        
        # Análisis de problemas
        self.log_info(f"\\n🔍 Análisis de Problemas:")
        
        if success_rate < 80:
            self.log_warning(f"⚠️ Tasa de éxito baja ({success_rate:.1f}%)")
            self.log_info("   Posibles problemas:")
            self.log_info("   - Threshold de la herramienta muy alto")
            self.log_info("   - Palabras clave no coinciden con consultas")
            self.log_info("   - Herramienta no filtra por cátedra específica")
        
        if 'router_sheets.horarios_secretarias' not in tool_usage:
            self.log_error("❌ La herramienta de secretarías no se activó en ninguna consulta")
        elif tool_usage.get('router_sheets.horarios_secretarias', 0) < total_queries * 0.7:
            self.log_warning("⚠️ La herramienta de secretarías se activó en menos del 70% de las consultas")
        
        # Verificar si las respuestas son específicas o genéricas
        if avg_response_length > 800:
            self.log_info("ℹ️ Las respuestas son muy largas - posiblemente devuelve todas las secretarías")
            self.log_info("   Recomendación: Implementar filtrado específico por cátedra")
        
        # Resultado del test
        if success_rate >= 80:
            self.log_success("🎉 TEST EXITOSO: La herramienta de secretarías funciona correctamente")
            return True
        elif success_rate >= 60:
            self.log_warning("⚠️ TEST PARCIALMENTE EXITOSO: La herramienta funciona pero necesita mejoras")
            return True
        else:
            self.log_error("❌ TEST FALLIDO: La herramienta de secretarías tiene problemas serios")
            return False


if __name__ == "__main__":
    test = TestSecretariasComprehensive()
    success = test.run_test()
    exit(0 if success else 1)
