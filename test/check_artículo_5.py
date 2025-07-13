import pandas as pd
import os
import re

# Ruta al archivo de metadatos
metadata_path = "data/embeddings/metadata.csv"

def check_article_5():
    print("🔍 Verificando la indexación del Artículo 5 del Régimen Disciplinario...")
    
    if not os.path.exists(metadata_path):
        print(f"❌ Error: No se encontró el archivo de metadatos en {metadata_path}")
        return
    
    try:
        metadata_df = pd.read_csv(metadata_path)
        print(f"✅ Metadatos cargados correctamente: {len(metadata_df)} registros")
        
        # Verificar si el Artículo 5 está completo en algún chunk
        art5_chunks = []
        for idx, row in metadata_df.iterrows():
            text = str(row.get('text', '')).lower()
            # Buscar patrones que coincidan con el Artículo 5
            if 'art. 5' in text or 'artículo 5' in text or 'art. 5º' in text or 'artículo 5º' in text:
                art5_chunks.append({
                    'index': idx,
                    'filename': row.get('filename', 'N/A'),
                    'text': str(row.get('text', 'N/A'))
                })
        
        print(f"\nSe encontraron {len(art5_chunks)} chunks que mencionan el Artículo 5")
        
        if art5_chunks:
            for i, chunk in enumerate(art5_chunks):
                print(f"\nChunk {i+1} (índice {chunk['index']}):")
                print(f"Archivo: {chunk['filename']}")
                print(f"Contenido:")
                print("="*80)
                print(chunk['text'])
                print("="*80)
                
                # Verificar si contiene la palabra "denuncia"
                if 'denuncia' in chunk['text'].lower():
                    print("✅ Este chunk contiene la palabra 'denuncia'")
                else:
                    print("⚠️ Este chunk NO contiene la palabra 'denuncia'")
                
                # Verificar si contiene información completa sobre presentación de denuncias
                if re.search(r'(presentar|presentación|formular).{0,50}denuncia', chunk['text'].lower()):
                    print("✅ Este chunk contiene información sobre cómo presentar denuncias")
                else:
                    print("⚠️ Este chunk NO parece contener información clara sobre cómo presentar denuncias")
        else:
            print("❌ No se encontró ningún chunk que mencione el Artículo 5")
            
        # Buscar chunks que contengan información sobre denuncias
        print("\n🔍 Buscando chunks que contengan la palabra 'denuncia'...")
        denuncia_chunks = []
        for idx, row in metadata_df.iterrows():
            text = str(row.get('text', '')).lower()
            if 'denuncia' in text:
                if not any(chunk['index'] == idx for chunk in art5_chunks):
                    denuncia_chunks.append({
                        'index': idx,
                        'filename': row.get('filename', 'N/A'),
                        'text': str(row.get('text', 'N/A'))
                    })
        
        print(f"\nSe encontraron {len(denuncia_chunks)} chunks adicionales que contienen la palabra 'denuncia'")
        if denuncia_chunks:
            for i, chunk in enumerate(denuncia_chunks):
                print(f"\nChunk adicional {i+1} (índice {chunk['index']}):")
                print(f"Archivo: {chunk['filename']}")
                print(f"Extracto relevante:")
                print("-"*80)
                text = chunk['text']
                # Mostrar contexto alrededor de la palabra denuncia
                for match in re.finditer(r'(.*\b)(denuncia\w*)(\b.*)', text, re.IGNORECASE):
                    context_before = match.group(1)[-100:] if len(match.group(1)) > 100 else match.group(1)
                    context_after = match.group(3)[:100] if len(match.group(3)) > 100 else match.group(3)
                    print(f"...{context_before}{match.group(2)}{context_after}...")
                print("-"*80)
                
        # Examinar cómo están divididos los chunks en el documento
        print("\n🔍 Analizando la división de chunks en el Régimen Disciplinario...")
        regimen_chunks = metadata_df[metadata_df['filename'] == 'Regimen_Disciplinario.pdf']
        print(f"El documento 'Regimen_Disciplinario.pdf' está dividido en {len(regimen_chunks)} chunks")
        
        # Mostrar el tamaño promedio de los chunks
        if not regimen_chunks.empty:
            avg_length = regimen_chunks['text_length'].mean()
            avg_words = regimen_chunks['word_count'].mean()
            print(f"Tamaño promedio de chunks: {avg_length:.2f} caracteres, {avg_words:.2f} palabras")
            
            # Verificar si hay continuidad entre chunks
            print("\nVerificando continuidad entre chunks...")
            for i in range(len(regimen_chunks) - 1):
                current_idx = regimen_chunks.iloc[i].name
                next_idx = regimen_chunks.iloc[i+1].name
                
                current_chunk = regimen_chunks.iloc[i]['text']
                next_chunk = regimen_chunks.iloc[i+1]['text']
                
                # Verificar si hay solapamiento o fragmentación
                current_end = current_chunk[-50:].lower()
                next_start = next_chunk[:50].lower()
                
                if any(phrase in next_start for phrase in ['art.', 'artículo']):
                    print(f"✅ El chunk {current_idx} termina y el {next_idx} comienza con un nuevo artículo")
                else:
                    print(f"⚠️ Posible fragmentación entre chunks {current_idx} y {next_idx}")
                    print(f"   Final de chunk {current_idx}: ...{current_end}")
                    print(f"   Inicio de chunk {next_idx}: {next_start}...")
        
    except Exception as e:
        print(f"❌ Error al procesar los metadatos: {str(e)}")
        
if __name__ == "__main__":
    check_article_5() 