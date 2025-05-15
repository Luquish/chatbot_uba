# Tests del Chatbot UBA

Este directorio contiene los archivos de prueba para el sistema de chatbot de la Facultad de Medicina UBA basado en RAG (Retrieval Augmented Generation).

## Scripts disponibles

### 1. Pruebas del índice FAISS
- **test_faiss.py**: Prueba básica del índice FAISS para verificar su carga correcta.
- **test_faiss_specific.py**: Prueba de consultas específicas en el índice FAISS.
- **check_artículo_5.py**: Verifica la correcta indexación del Artículo 5 del Régimen Disciplinario.

### 2. Pruebas de consultas y respuestas
- **test_denuncia_query.py**: Prueba específica para consultas sobre denuncias.
- **test_normativas.py**: Conjunto de 20 preguntas sobre Condiciones de Regularidad y Régimen Disciplinario.

## Cómo ejecutar los tests

Para ejecutar cualquiera de los tests, utiliza:

```bash
python test/nombre_del_test.py
```

Por ejemplo:
```bash
python test/test_normativas.py
```

## Notas importantes

- Por defecto, los tests están configurados para no realizar llamadas a la API de OpenAI cuando sea posible, para evitar costos innecesarios. Para ejecutar consultas reales, descomenta las secciones correspondientes en los scripts.

- Para que los tests funcionen correctamente, asegúrate de tener las variables de entorno adecuadas (especialmente OPENAI_API_KEY) configuradas en un archivo .env en la raíz del proyecto.

- Los tests asumen que el índice FAISS y los metadatos están correctamente generados en la carpeta data/embeddings/. 