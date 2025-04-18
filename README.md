# UBA Administrative Chatbot Template

A template for building educational chatbots for administrative support at any faculty of the University of Buenos Aires (UBA). This project implements a WhatsApp-based chatbot that uses RAG (Retrieval Augmented Generation) to provide accurate and contextual responses to administrative queries.

## Project Overview

This project aims to develop a template for creating administrative support chatbots for any faculty at the University of Buenos Aires. At its core, it's a RAG (Retrieval-Augmented Generation) system that integrates the following functionalities:

### Document Preprocessing
- Extracts and cleans text from PDF documents (e.g., regularity conditions, disciplinary regulations)
- Segments text into chunks with overlap to maintain context
- Generates metadata for each chunk
- Easily adaptable to any faculty's specific documentation

### Embeddings Generation and Context Retrieval
- Uses multilingual models (like sentence-transformers) to transform chunks into embeddings
- Indexes locally with FAISS for testing
- Can be configured to use Pinecone in production
- Enables relevant information retrieval in response to student queries
- Supports both Spanish and local Argentine expressions

### Modeling and Fine-Tuning
- Works with open-source LLMs (like Mistral or lighter alternatives based on resource availability)
- Implements LoRA fine-tuning to adapt the model to:
  - Administrative context and language
  - Interaction examples
  - Local expressions (Argentine Spanish)
  - Faculty-specific terminology

### Backend and API
- Built with FastAPI for asynchronous and scalable request handling
- Direct integration with WhatsApp Cloud API for messaging
- Robust error handling and logging
- Easy to configure and deploy

### System Integration
The complete flow works as follows:
1. Student sends a query via WhatsApp
2. WhatsApp Cloud API forwards the message to our FastAPI backend
3. RAG system queries the embeddings index to retrieve relevant context from processed documents
4. Fine-tuned LLM generates a response combining the context and query
5. Response is sent back to the student through WhatsApp

### Key Features
- WhatsApp Business API integration
- RAG-based response generation
- Fine-tuned language model
- Automatic setup and deployment
- Robust error handling
- Detailed logging
- Easy customization for different faculties

This template provides a reliable and scalable tool that combines natural language processing, information retrieval, and messaging (through WhatsApp) to provide administrative support to UBA students, adaptable to any faculty's specific needs and documentation.

## Features

- WhatsApp Business API integration
- RAG-based response generation
- Fine-tuned language model
- Automatic setup and deployment
- Robust error handling
- Detailed logging

## Project Structure

```
.
├── data/
│   ├── raw/           # Raw data files
│   ├── processed/     # Processed data
│   └── embeddings/    # Generated embeddings
├── models/            # Model files
├── scripts/           # Python scripts
│   ├── auto_setup.py  # Automatic setup
│   ├── create_embeddings.py  # Embedding generation
│   ├── deploy_backend.py     # Backend deployment
│   ├── preprocess.py  # Data preprocessing
│   ├── run_rag.py     # RAG system
│   └── train_finetune.py  # Model fine-tuning
└── logs/              # Log files
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatbot_uba.git
cd chatbot_uba
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```
ENVIRONMENT=development
MODEL_PATH=models/finetuned_model
EMBEDDINGS_DIR=data/embeddings
WHATSAPP_API_TOKEN=your_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_account_id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_verify_token
MY_PHONE_NUMBER=your_test_number
HUGGING_FACE_HUB_TOKEN=your_huggingface_token  # Required for model inference
```

4. Run the auto setup script:
```bash
python scripts/auto_setup.py
```

This script will:
- Start the backend server (deploy_backend.py)
- Configure ngrok to create a public URL
- Verify WhatsApp token validity
- Display instructions for Glitch configuration
- Keep services running for development
- Handle graceful shutdown on Ctrl+C

The script will show you:
- The local backend URL (http://localhost:8000)
- The ngrok public URL for Glitch configuration
- Instructions for setting up the webhook in Meta
- Instructions for configuring Glitch variables

Keep this script running during development to maintain the backend and ngrok tunnel active.

## Development

1. Start the backend server:
```

## Scripts

### preprocess.py

Specialized script for preprocessing PDF documents for the RAG system.

#### Main Features:

1. **Text Extraction**
   - Uses pdfminer for robust text extraction
   - Fallback system for problematic documents
   - Document structure preservation
   - Handles complex university administrative documents

2. **University Document Processing**
   - Recognition of specific sections (articles, resolutions, etc.)
   - Conversion to structured Markdown format
   - Document hierarchy preservation
   - Special handling of regulatory and administrative content

3. **Intelligent Chunking System**
   - Text division into RAG-optimized fragments
   - Context preservation in each chunk
   - Special handling of articles and sections
   - Configurable overlap between chunks
   - Smart chunk size management

4. **Data Management**
   - Metadata generation for each document
   - Detailed logging system
   - CSV results export
   - Individual chunk storage
   - Document processing tracking

#### Usage:

```bash
python scripts/preprocess.py
```

#### Directory Structure:
- Input: `data/raw/` - Raw PDF documents
- Output: 
  - `data/processed/` - Processed documents in Markdown format
  - `data/processed/chunks/` - Individual chunks
  - `data/processed/processed_documents.csv` - Processing metadata

#### Configuration:
- Chunk size: 250 words (configurable)
- Overlap: 50 words (configurable)
- Minimum chunk size: 50 words

#### Key Methods:
- `extract_text_from_pdf`: Robust PDF text extraction
- `clean_text`: Text normalization and cleaning
- `convert_to_markdown`: Conversion to structured Markdown
- `split_into_chunks`: Intelligent document chunking
- `process_document`: Complete document processing pipeline

### create_embeddings.py

Script responsible for generating and managing text embeddings using OpenAI's API.

#### Main Features:

1. **OpenAI Integration**
   - Uses OpenAI's text-embedding models (text-embedding-3-small)
   - Robust API error handling
   - Configurable batch processing
   - Automatic rate limiting management

2. **Embedding Generation**
   - Processes chunks from preprocessed documents
   - Handles batch processing for efficiency
   - Implements automatic retries and error recovery
   - Normalizes embeddings for consistent similarity search

3. **Vector Storage**
   - Creates and manages FAISS indices for vector storage
   - Optimizes index type based on dataset size
   - Implements efficient similarity search capabilities
   - Handles both small and large-scale vector collections

4. **Data Management**
   - Comprehensive metadata tracking
   - Detailed logging of embedding generation process
   - Statistical summaries of processed documents
   - Configuration tracking and versioning

#### Usage:

```bash
python scripts/create_embeddings.py
```

#### Requirements:
- OpenAI API key configured in `.env`
- Processed documents in `data/processed/`
- Sufficient disk space for embedding storage

#### Directory Structure:
- Input: `data/processed/processed_documents.csv`
- Output:
  - `data/embeddings/faiss_index.bin` - FAISS vector index
  - `data/embeddings/metadata.csv` - Detailed chunk metadata
  - `data/embeddings/metadata_summary.csv` - Processing statistics
  - `data/embeddings/config.json` - Configuration details

#### Key Methods:
- `generate_embeddings`: Creates embeddings using OpenAI's API
- `create_faiss_index`: Builds optimized FAISS indices
- `save_metadata`: Stores comprehensive embedding metadata
- `process_documents`: Main pipeline for embedding generation

#### Configuration:
- Embedding model: text-embedding-3-small (default)
- Batch size: 16 (configurable)
- Vector dimension: 1536 (small) or 3072 (large)
- Automatic index optimization for collections > 10,000 vectors

# Chatbot UBA - Procesamiento de Documentos

Este repositorio contiene un sistema de procesamiento de documentos para el chatbot de la Universidad de Buenos Aires, que incluye las siguientes funcionalidades:

## Procesamiento de Documentos con Marker PDF

El sistema utiliza [Marker PDF](https://github.com/VikParuchuri/marker), una herramienta avanzada para convertir documentos a markdown, JSON y HTML con alta precisión.

### Características implementadas

- **Conversión de PDF a Markdown**: Procesa documentos PDF con alta precisión, preservando la estructura original.
- **Extracción de texto inteligente**: Utiliza OCR cuando es necesario para mejorar la calidad del texto extraído.
- **Detección de estructura**: Mantiene la jerarquía de los documentos (títulos, secciones, artículos, etc.).
- **División en chunks**: Segmenta los documentos en fragmentos de tamaño adecuado para su posterior indexación.
- **Procesamiento de tablas**: Reconoce y formatea correctamente las tablas en el texto.
- **Manejo de paginación**: Preserva información de paginación para facilitar referencias al documento original.

### Instalación

Para usar el sistema de procesamiento, instale las dependencias:

```bash
pip install marker-pdf pandas tqdm
```

### Uso

1. Coloque los archivos PDF a procesar en la carpeta `data/raw/`
2. Ejecute el script de preprocesamiento:

```bash
python -m scripts.preprocess
```

3. Los resultados se guardarán en `data/processed/`:
   - Archivos Markdown para cada documento
   - Archivos de texto para cada chunk
   - CSV con metadatos de los documentos procesados
   - Imágenes extraídas (si las hay)

### Estructura

El procesamiento se realiza mediante la clase `MarkerPreprocessor` que:

1. Detecta y verifica la instalación de Marker
2. Procesa cada documento PDF mediante `marker_single`
3. Divide el texto en chunks respetando la estructura jerárquica
4. Guarda los resultados en formatos apropiados para su uso en el sistema RAG

### Ventajas sobre otros procesadores

- Alta precisión en la extracción de texto
- Excelente detección de la estructura del documento
- Conversión precisa de tablas y elementos complejos
- Mejor preservación del formato original
- Capacidad de utilizar OCR avanzado para documentos escaneados

### Integración con el sistema RAG

Los chunks generados están optimizados para su uso con sistemas de Recuperación Aumentada por Generación (RAG), incluyendo:

- Prefijos con metadatos del documento de origen
- Tamaño adecuado para indexación
- Preservación de contexto en la división (manteniendo títulos y referencias)
- Formato consistente para mejorar el retrieval

## Próximos pasos

- Implementar soporte para documentos DOCX y otros formatos
- Agregar extracción de metadatos avanzados
- Mejorar la detección de elementos específicos para documentos de la UBA
