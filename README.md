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