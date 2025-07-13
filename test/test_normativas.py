import os
import sys
from dotenv import load_dotenv

# Añadir el directorio raíz al path para poder importar rag_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_system import RAGSystem

"""
Este script prueba la capacidad del sistema RAG para responder
consultas relacionadas con las Condiciones de Regularidad y el
Régimen Disciplinario de la Universidad.
"""

def test_preguntas_normativas():
    """Prueba preguntas sobre normativas y reglamentos universitarios"""
    print("🔍 Iniciando pruebas de consultas sobre normativas...")
    
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar la API key de OpenAI
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ No se encontró la API key de OpenAI. Por favor configura la variable OPENAI_API_KEY.")
        return False
    
    try:
        # Inicializar el sistema RAG
        print("Inicializando sistema RAG...")
        rag_system = RAGSystem()
        print("✅ Sistema RAG inicializado correctamente")
        
        # Lista de 20 preguntas sobre las normativas
        preguntas = [
            # Preguntas sobre Condiciones de Regularidad
            "¿Cuáles son las condiciones para mantener la regularidad en la UBA?",
            "¿Cuántas materias debo aprobar para mantener la regularidad?",
            "¿Cuál es el porcentaje máximo de aplazos permitido?",
            "¿En cuánto tiempo debo completar mi carrera para mantener la regularidad?",
            "¿Qué pasa si pierdo la condición de alumno regular?",
            "¿Cómo puedo solicitar la readmisión si perdí la regularidad?",
            "¿Cuáles son las causales para ser readmitido automáticamente?",
            "¿Qué es la Comisión de Readmisión y quiénes la integran?",
            "¿Puedo suspender mi condición de alumno temporalmente?",
            "¿Qué ocurre con las materias que ya aprobé si pierdo la regularidad?",
            
            # Preguntas sobre Régimen Disciplinario
            "¿Qué tipos de sanciones existen para los estudiantes?",
            "¿Por cuánto tiempo pueden suspender a un alumno?",
            "¿Qué ocurre si me falto el respeto a un profesor?",
            "¿Cuál es el procedimiento para presentar una denuncia contra un estudiante?",
            "¿En qué casos se aplica una suspensión de más de 5 años?",
            "¿Quién puede aplicar sanciones a los estudiantes?",
            "¿Puedo apelar una sanción disciplinaria?",
            "¿Cuándo prescribe una falta disciplinaria?",
            "¿Qué implica estar suspendido en la Universidad?",
            "¿Puede la Universidad iniciar un sumario de oficio?"
        ]
        
        # Probar cada pregunta
        for i, pregunta in enumerate(preguntas):
            print(f"\n{'='*80}")
            print(f"Pregunta {i+1}: '{pregunta}'")
            print(f"{'='*80}")
            
            # Procesar la consulta
            print("Procesando consulta...")
            
            # Ejecutar la consulta real
            result = rag_system.process_query(pregunta)
            
            print(f"\nRESPUESTA GENERADA:")
            print(f"{'-'*80}")
            print(result.get("response", ""))
            print(f"{'-'*80}")
            
            print(f"\nFUENTES:")
            for source in result.get("sources", []):
                print(f"- {source}")
            
        print("\n✅ Todas las preguntas de prueba han sido ejecutadas correctamente.")
        return True
        
    except Exception as e:
        print(f"❌ Error al probar las consultas: {str(e)}")
        return False

if __name__ == "__main__":
    test_preguntas_normativas() 