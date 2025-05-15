import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
import json

# Configurar rutas
EMBEDDINGS_DIR = "data/embeddings"

def test_faiss_index():
    print("🔍 Probando índice FAISS...")
    
    # Cargar el índice FAISS
    index_path = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
    if not os.path.exists(index_path):
        print(f"❌ Error: No se encontró el archivo de índice FAISS en {index_path}")
        return False
    
    try:
        index = faiss.read_index(index_path)
        print(f"✅ Índice FAISS cargado correctamente")
        print(f"   Dimensiones del índice: {index.d}")
        print(f"   Número de vectores: {index.ntotal}")
    except Exception as e:
        print(f"❌ Error al cargar el índice FAISS: {str(e)}")
        return False
    
    # Cargar los metadatos
    metadata_path = os.path.join(EMBEDDINGS_DIR, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"❌ Error: No se encontró el archivo de metadatos en {metadata_path}")
        return False
    
    try:
        metadata_df = pd.read_csv(metadata_path)
        print(f"✅ Metadatos cargados correctamente")
        print(f"   Número de registros en metadata: {len(metadata_df)}")
        
        # Mostrar primeras entradas para verificar estructura
        print("\nPrimeros registros de metadatos:")
        print(metadata_df.head(2).to_string())
    except Exception as e:
        print(f"❌ Error al cargar los metadatos: {str(e)}")
        return False
    
    # Intentar hacer una consulta de prueba
    try:
        # Crear un vector de prueba aleatorio con la misma dimensión que el índice
        test_query = np.random.random(index.d).astype('float32')
        test_query = np.expand_dims(test_query, axis=0)
        
        # Búsqueda
        k = 3  # Número de resultados a devolver
        distances, indices = index.search(test_query, k)
        
        print("\n✅ Búsqueda de prueba realizada correctamente")
        print(f"   Distancias: {distances[0]}")
        print(f"   Índices recuperados: {indices[0]}")
        
        # Mostrar los metadatos correspondientes a los resultados
        print("\nMetadatos de los resultados:")
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata_df) and idx >= 0:
                filename = metadata_df.iloc[idx].get('filename', 'N/A')
                text_preview = str(metadata_df.iloc[idx].get('text', 'N/A'))
                if len(text_preview) > 100:
                    text_preview = text_preview[:100] + "..."
                print(f"\nResultado {i+1} (índice {idx}):")
                print(f"Archivo: {filename}")
                print(f"Texto: {text_preview}")
            else:
                print(f"⚠️ Índice fuera de rango: {idx}")
    
    except Exception as e:
        print(f"❌ Error al realizar la búsqueda de prueba: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    load_dotenv()
    success = test_faiss_index()
    if success:
        print("\n✅ El índice FAISS parece estar funcionando correctamente")
    else:
        print("\n❌ Se encontraron problemas con el índice FAISS") 