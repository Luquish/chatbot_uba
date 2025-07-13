import sys
from pathlib import Path

# Añadir el directorio raíz al path para acceder a rag_system.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system import RAGSystem


def main():
    """Función principal para ejecutar el sistema RAG."""
    print("\nBienvenido al sistema RAG de DrCecim")
    print("Este sistema usa modelos de OpenAI para responder consultas sobre la Facultad de Medicina UBA")
    print("Escribe 'salir' para terminar")

    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"Error al inicializar el sistema: {str(e)}")
        return
    
    while True:
        query = input("\nIngrese su consulta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
        try:
            result = rag.process_query(query)
            print("\nRespuesta:", result['response'])
            if result.get('sources'):
                print("\nFuentes consultadas:")
                for source in result['sources']:
                    print(f"- {source}")
        except Exception as e:
            print("Lo siento, hubo un error al procesar tu consulta.")


if __name__ == "__main__":
    main() 