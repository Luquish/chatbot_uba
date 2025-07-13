import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
import json
from unidecode import unidecode
import re

# Configurar rutas
EMBEDDINGS_DIR = "data/embeddings"

def normalize_text(text):
    """Normaliza el texto eliminando tildes y caracteres especiales"""
    normalized = text.lower().strip()
    normalized = unidecode(normalized)  # Eliminar tildes
    normalized = re.sub(r'[^\w\s]', '', normalized)  # Eliminar signos de puntuación
    normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalizar espacios
    return normalized

def test_query(query):
    """Prueba una consulta específica en el índice FAISS"""
    print(f"🔍 Probando consulta: '{query}'")
    
    # Cargar el índice FAISS
    index_path = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
    if not os.path.exists(index_path):
        print(f"❌ Error: No se encontró el archivo de índice FAISS en {index_path}")
        return False
    
    try:
        index = faiss.read_index(index_path)
        print(f"✅ Índice FAISS cargado correctamente")
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
        print(f"✅ Metadatos cargados correctamente: {len(metadata_df)} registros")
    except Exception as e:
        print(f"❌ Error al cargar los metadatos: {str(e)}")
        return False
    
    # Verificar si hay chunks que contengan la palabra "denuncia"
    try:
        # Buscar chunks con la palabra denuncia
        print("\n🔍 Verificando chunks con la palabra 'denuncia'...")
        denuncia_chunks = []
        for idx, row in metadata_df.iterrows():
            if 'text' in row and 'denuncia' in str(row['text']).lower():
                denuncia_chunks.append({
                    'index': idx,
                    'filename': row.get('filename', 'N/A'),
                    'text': str(row.get('text', 'N/A'))[:150] + "..." if len(str(row.get('text', 'N/A'))) > 150 else str(row.get('text', 'N/A'))
                })
        
        print(f"✅ Se encontraron {len(denuncia_chunks)} chunks que contienen la palabra 'denuncia'")
        if denuncia_chunks:
            print("\nPrimeros 3 chunks con la palabra 'denuncia':")
            for i, chunk in enumerate(denuncia_chunks[:3]):
                print(f"\nChunk {i+1} (índice {chunk['index']}):")
                print(f"Archivo: {chunk['filename']}")
                print(f"Texto: {chunk['text']}")
        else:
            print("⚠️ No se encontraron chunks con la palabra 'denuncia'")
    
    except Exception as e:
        print(f"❌ Error al buscar chunks con la palabra 'denuncia': {str(e)}")
    
    # Buscar chunks con palabras relacionadas (denunciar, denuncias)
    print("\n🔍 Verificando chunks con palabras relacionadas (denunciar, denuncias)...")
    related_chunks = []
    for idx, row in metadata_df.iterrows():
        text = str(row.get('text', '')).lower()
        if 'denunciar' in text or 'denuncias' in text:
            if not any(chunk['index'] == idx for chunk in denuncia_chunks):
                related_chunks.append({
                    'index': idx,
                    'filename': row.get('filename', 'N/A'),
                    'text': text[:150] + "..." if len(text) > 150 else text
                })
    
    print(f"✅ Se encontraron {len(related_chunks)} chunks adicionales con palabras relacionadas")
    if related_chunks:
        print("\nChunks con palabras relacionadas:")
        for i, chunk in enumerate(related_chunks[:3]):
            print(f"\nChunk adicional {i+1} (índice {chunk['index']}):")
            print(f"Archivo: {chunk['filename']}")
            print(f"Texto: {chunk['text']}")
    
    # Verificar el Artículo 5º del Régimen Disciplinario específicamente
    print("\n🔍 Buscando el Artículo 5º del Régimen Disciplinario...")
    art5_chunks = []
    for idx, row in metadata_df.iterrows():
        text = str(row.get('text', '')).lower()
        if ('art. 5' in text or 'artículo 5' in text or 'articulo 5' in text) and 'regimen disciplinario' in text:
            art5_chunks.append({
                'index': idx,
                'filename': row.get('filename', 'N/A'),
                'text': str(row.get('text', 'N/A'))
            })
    
    if art5_chunks:
        print(f"✅ Se encontró el Artículo 5º en {len(art5_chunks)} chunks")
        for i, chunk in enumerate(art5_chunks):
            print(f"\nChunk con Artículo 5º (índice {chunk['index']}):")
            print(f"Archivo: {chunk['filename']}")
            print(f"Texto: {chunk['text']}")
    else:
        print("⚠️ No se encontró el Artículo 5º del Régimen Disciplinario")
    
    # Ahora realicemos una simulación de búsqueda semántica
    try:
        # Para una prueba real, necesitaríamos un modelo de embedding.
        # En este caso, usaremos un vector aleatorio como simulación
        test_query = np.random.random(index.d).astype('float32')
        test_query = np.expand_dims(test_query, axis=0)
        
        k = 5  # Número de resultados a devolver
        distances, indices = index.search(test_query, k)
        
        print("\n✅ Simulación de búsqueda semántica realizada")
        print(f"   Resultados para una consulta aleatoria (simulada):")
        print(f"   Índices recuperados: {indices[0]}")
    except Exception as e:
        print(f"❌ Error al realizar la simulación de búsqueda: {str(e)}")
    
    return True

if __name__ == "__main__":
    load_dotenv()
    query = "como presento una denuncia"
    test_query(query) 