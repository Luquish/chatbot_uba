# Chatbot Administrativo UBA

Un sistema de chatbot administrativo para la Universidad de Buenos Aires (UBA) basado en tecnología RAG (Retrieval Augmented Generation) e integración con WhatsApp. El sistema proporciona respuestas precisas y contextuales a consultas administrativas utilizando documentos institucionales.

## Descripción General

Este proyecto implementa un asistente virtual para cualquier facultad de la UBA que responde consultas administrativas a través de WhatsApp. El sistema utiliza técnicas avanzadas de procesamiento de lenguaje natural:

### Características Principales

- **Procesamiento de Documentos**: Extrae y procesa texto de documentos PDF utilizando Marker PDF, preservando la estructura y generando chunks optimizados para RAG.
- **Sistema de Embeddings**: Utiliza OpenAI (text-embedding-3-small) para generar embeddings de alta calidad y almacenarlos en índices FAISS.
- **Motor RAG**: Sistema de Generación Aumentada por Recuperación que combina recuperación de información y generación de respuestas contextuales.
- **Integración WhatsApp**: Conexión directa con la API de WhatsApp Business para enviar y recibir mensajes.
- **Sistema de Intenciones**: Clasificación semántica de consultas para personalizar las respuestas según el tipo de pregunta.
- **Fine-tuning de Modelos**: Capacidad de fine-tuning mediante OpenAI para adaptar el modelo a interacciones específicas.

### Flujo del Sistema

1. El estudiante envía una consulta por WhatsApp
2. El backend de FastAPI recibe el mensaje a través de un webhook
3. El sistema RAG analiza la consulta e identifica la intención
4. Se recupera información relevante desde la base de conocimientos
5. Se genera una respuesta utilizando la información contextual y la consulta
6. La respuesta se envía de vuelta al estudiante a través de WhatsApp

## Estructura del Proyecto

```
.
├── data/
│   ├── raw/           # Documentos PDF sin procesar
│   ├── processed/     # Documentos procesados en formato markdown
│   ├── embeddings/    # Embeddings e índices FAISS generados
│   └── finetuning/    # Datos para fine-tuning del modelo
├── models/            # Modelos entrenados o fine-tuneados
├── scripts/
│   ├── preprocess.py        # Procesamiento de documentos PDF con Marker
│   ├── create_embeddings.py # Generación de embeddings con OpenAI
│   ├── run_rag.py           # Sistema RAG para consultas y respuestas
│   ├── deploy_backend.py    # Backend FastAPI y webhook de WhatsApp
│   ├── train_finetune.py    # Fine-tuning de modelos con OpenAI
│   └── auto_setup.py        # Configuración automática del entorno
├── logs/              # Archivos de registro
└── config/            # Archivos de configuración
```

## Configuración

1. Clona el repositorio:
```bash
git clone https://github.com/yourusername/chatbot_uba.git
cd chatbot_uba
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Crea un archivo `.env` basado en el template `.env.example`:
```bash
cp .env.example .env
```

4. Configura las variables de entorno en el archivo `.env`:

### Variables de Entorno Principales

```
# General
ENVIRONMENT=development                   # development o production

# WhatsApp Business API
WHATSAPP_API_TOKEN=your_token             # Token de la API de WhatsApp
WHATSAPP_PHONE_NUMBER_ID=your_phone_id    # ID del número de teléfono en WhatsApp Business
WHATSAPP_BUSINESS_ACCOUNT_ID=your_account_id  # ID de la cuenta de negocio
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_token  # Token para verificar webhooks

# Configuración de Glitch (para webhook en desarrollo)
GLITCH_PROJECT_NAME=your_project_name     # Nombre del proyecto en Glitch
GLITCH_API_URL=https://api.glitch.com     # URL de la API de Glitch

# Backend y servidor
BACKEND_URL=http://localhost:8000/api/whatsapp/message  # URL para webhooks
HOST=0.0.0.0                              # Host para el servidor FastAPI
PORT=8000                                 # Puerto para el servidor FastAPI

# Configuración de OpenAI
OPENAI_API_KEY=your_openai_key            # API Key de OpenAI
PRIMARY_MODEL=gpt-4o-mini                 # Modelo principal de OpenAI
FALLBACK_MODEL=gpt-4.1-nano               # Modelo de respaldo si falla el principal
EMBEDDING_MODEL=text-embedding-3-small    # Modelo para embeddings


# Parámetros de generación
MAX_LENGTH=512                            # Longitud máxima de contexto
TEMPERATURE=0.7                           # Temperatura para generación
TOP_P=0.9                                 # Parámetro top_p para generación
TOP_K=50                                  # Parámetro top_k para generación
MAX_OUTPUT_TOKENS=300                     # Máxima longitud de salida

# Configuración de RAG
RAG_NUM_CHUNKS=3                          # Número de chunks a recuperar
SIMILARITY_THRESHOLD=0.3                  # Umbral de similitud mínima

# Directorios de datos
MODEL_PATH=models/finetuned_model         # Directorio para modelos
EMBEDDINGS_DIR=data/embeddings            # Directorio para embeddings

# Dispositivo para cálculos
DEVICE=mps                                # auto, cuda, cpu, o mps para Mac
```

5. Ejecuta el script de configuración automática:
```bash
python scripts/auto_setup.py
```

## Flujo de Trabajo

### Desarrollo Local

1. **Procesamiento de Documentos:**
```bash
python scripts/preprocess.py
```
Este script:
- Procesa los PDFs en `data/raw/`
- Genera chunks optimizados
- Guarda los resultados en `data/processed/`

2. **Generación de Embeddings:**
```bash
python scripts/create_embeddings.py
```
Este script:
- Genera embeddings usando OpenAI
- Crea y guarda el índice FAISS
- Almacena metadatos necesarios

3. **Ejecución Local:**
```bash
python scripts/deploy_backend.py
```

### Despliegue con Docker

El proyecto está configurado para ser desplegado usando Docker, separando el procesamiento de documentos (local) de la ejecución del servidor (producción).

#### Estructura Docker

```
.
├── Dockerfile           # Configuración de la imagen
├── docker-compose.yml   # Configuración del servicio
└── .dockerignore       # Archivos excluidos del contenedor
```

#### Archivos en Producción

Solo los archivos necesarios para la ejecución se incluyen en el contenedor:
- `scripts/deploy_backend.py`
- `scripts/run_rag.py`
- `scripts/calendar_service.py`
- `scripts/date_utils.py`
- `data/embeddings/`
- `config/calendar_config.py`

#### Comandos Docker

1. **Construir la imagen:**
```bash
docker-compose build
```

2. **Iniciar el contenedor:**
```bash
docker-compose up -d
```

3. **Ver logs:**
```bash
docker-compose logs -f
```

4. **Detener el contenedor:**
```bash
docker-compose down
```

#### Variables de Entorno para Producción

Crear un archivo `.env` con:
```env
# Configuración del entorno
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000

# WhatsApp Business API
WHATSAPP_API_TOKEN=your_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_id_here
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id_here
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_webhook_token_here

# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Configuración del modelo
PRIMARY_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### Flujo de Actualización de Documentos

1. Añadir nuevos PDFs en `data/raw/`
2. Ejecutar localmente el preprocesamiento:
   ```bash
   python scripts/preprocess.py
   python scripts/create_embeddings.py
   ```
3. Los nuevos embeddings se generarán en `data/embeddings/`
4. El contenedor Docker montará automáticamente los nuevos embeddings

### Endpoints Disponibles

- `GET /health`: Estado del servicio
- `POST /chat`: Endpoint para consultas directas
- `POST /webhook/whatsapp`: Webhook para mensajes de WhatsApp
- `GET /test-message`: Envía mensaje de prueba

### Monitoreo y Mantenimiento

- Los logs se encuentran en la carpeta `logs/`
- El healthcheck verifica el servicio cada 30 segundos
- El contenedor se reinicia automáticamente en caso de fallo

## Requisitos

- Python 3.9+
- Marker PDF para procesamiento de documentos
- Cuenta de OpenAI con API key
- Cuenta de WhatsApp Business
- ngrok para desarrollo local

## Dependencias Principales

- FastAPI y Uvicorn para el backend
- Transformers y PyTorch para modelos de NLP
- FAISS para índices de vectores
- OpenAI para embeddings y fine-tuning
- Marker PDF para procesamiento de documentos

## Licencia

Este proyecto está bajo licencia [Incluir la licencia correspondiente]
