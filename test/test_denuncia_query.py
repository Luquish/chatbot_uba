import os
import sys
from dotenv import load_dotenv
from rag_system import RAGSystem

def test_denuncia_query():
    """Prueba específica para consultas sobre denuncias"""
    print("🔍 Iniciando prueba de consulta sobre denuncias...")
    
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
        
        # Consultas a probar
        denuncias_queries = [
            "como presento una denuncia",
            "dónde puedo presentar una denuncia",
            "cuál es el procedimiento para denunciar",
            "quiero hacer una denuncia, qué debo hacer"
        ]
        
        # Probar cada consulta
        for i, query in enumerate(denuncias_queries):
            print(f"\n{'='*80}")
            print(f"Consulta {i+1}: '{query}'")
            print(f"{'='*80}")
            
            # Procesar la consulta
            result = rag_system.process_query(query)
            
            # Verificar si la respuesta contiene información relevante sobre denuncias
            response = result.get("response", "")
            denuncia_keywords = ["denuncia", "denuncias", "por escrito", "escrita", "48 horas", "artículo 5"]
            
            has_relevant_info = any(keyword in response.lower() for keyword in denuncia_keywords)
            
            # Mostrar información sobre la respuesta
            print(f"\nRESPUESTA GENERADA:")
            print(f"{'-'*80}")
            print(response)
            print(f"{'-'*80}")
            
            # Mostrar información sobre los chunks recuperados
            print(f"\nCHUNKS RECUPERADOS:")
            for j, chunk in enumerate(result.get("relevant_chunks", [])):
                filename = chunk.get("filename", "desconocido")
                similarity = chunk.get("similarity", 0)
                text_preview = chunk.get("text", "")[:150] + "..." if len(chunk.get("text", "")) > 150 else chunk.get("text", "")
                print(f"Chunk {j+1}: {filename} (similitud: {similarity:.2f})")
                print(f"Texto: {text_preview}")
                print()
            
            # Verificar si la respuesta contiene la información adecuada
            if "Art. 5" in response or "Artículo 5" in response:
                print("✅ La respuesta incluye referencia al Artículo 5 del Régimen Disciplinario")
            else:
                print("⚠️ La respuesta no incluye referencia al Artículo 5 del Régimen Disciplinario")
                
            if "por escrito" in response.lower():
                print("✅ La respuesta menciona que las denuncias deben presentarse por escrito")
            else:
                print("⚠️ La respuesta no menciona que las denuncias deben presentarse por escrito")
                
            if "48" in response:
                print("✅ La respuesta menciona el plazo de 48 horas para ratificar denuncias verbales")
            else:
                print("⚠️ La respuesta no menciona el plazo de 48 horas")
            
            if has_relevant_info:
                print("✅ La respuesta contiene información relevante sobre denuncias")
            else:
                print("❌ La respuesta no contiene información relevante sobre denuncias")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al probar la consulta sobre denuncias: {str(e)}")
        return False

if __name__ == "__main__":
    test_denuncia_query() 