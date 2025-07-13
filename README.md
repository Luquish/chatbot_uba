# Chatbot Administrativo UBA

Un sistema de chatbot administrativo para la Universidad de Buenos Aires (UBA) basado en tecnología RAG (Retrieval Augmented Generation) e integración con WhatsApp. El sistema proporciona respuestas precisas y contextuales a consultas administrativas utilizando documentos institucionales.

## Descripción General

Este proyecto implementa un asistente virtual para cualquier facultad de la UBA que responde consultas administrativas a través de WhatsApp. El sistema utiliza técnicas avanzadas de procesamiento de lenguaje natural:

### Características Principales

- **Sistema RAG**: Generación Aumentada por Recuperación que combina recuperación de información y generación de respuestas contextuales.
- **Integración WhatsApp**: Conexión directa con la API de WhatsApp Business para enviar y recibir mensajes.
- **Sistema de Intenciones**: Clasificación semántica de consultas para personalizar las respuestas según el tipo de pregunta.
- **Almacenamiento Híbrido**: Carga embeddings desde Google Cloud Storage con fallback local.
- **Integración con Google APIs**: Calendario y Sheets para información dinámica.

### Flujo del Sistema

1. El estudiante envía una consulta por WhatsApp
2. El backend de FastAPI recibe el mensaje a través de un webhook
3. El sistema RAG analiza la consulta e identifica la intención
4. Se recupera información relevante desde la base de conocimientos
5. Se genera una respuesta utilizando la información contextual y la consulta
6. La respuesta se envía de vuelta al estudiante a través de WhatsApp

## Estructura del Proyecto

```
chatbot_uba/                     # Backend del chatbot
├── main.py                      # FastAPI + WhatsApp webhook
├── rag_system.py               # Sistema RAG principal
├── requirements.txt            # Dependencias (dev + prod unificadas)
├── Dockerfile                  # Imagen de contenedor
├── docker-compose.yml          # Configuración base
├── docker-compose.override.yml # Volúmenes para desarrollo
├── data/
│   └── embeddings/            # Fallback local para embeddings
├── config/                    # Configuraciones del sistema
├── models/                    # Modelos OpenAI
├── storage/                   # Vector store (FAISS)
├── handlers/                  # Manejadores de intenciones
├── services/                  # Servicios externos (Calendar, Sheets)
├── utils/                     # Utilidades del sistema
└── scripts/
    ├── run_rag.py            # Testing del RAG
    └── gcs_storage.py        # Integración con GCS
```

## Configuración

1. Clona el repositorio:
```bash
git clone https://github.com/yourusername/chatbot_uba.git
cd chatbot_uba
```

2. Crea y activa un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Configura las variables de entorno en `.env`:

### Variables de Entorno Principales

```env
# OpenAI
OPENAI_API_KEY=your-openai-api-key
PRIMARY_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# WhatsApp Business API
WHATSAPP_API_TOKEN=your-whatsapp-token
WHATSAPP_PHONE_NUMBER_ID=your-phone-number-id
WHATSAPP_BUSINESS_ACCOUNT_ID=your-business-account-id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your-verify-token

# Google Cloud Storage (opcional)
USE_GCS=true
GCS_BUCKET_NAME=your-bucket-name

# Google APIs (opcional)
GOOGLE_API_KEY=your-google-api-key
CURSOS_SPREADSHEET_ID=your-spreadsheet-id

# Configuración del servidor
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8080
```

## Despliegue

### Desarrollo Local

```bash
# Ejecutar directamente
uvicorn main:app --reload --port 8080

# Probar el sistema RAG
python -m scripts.run_rag
```

### Docker (Recomendado)

```bash
# Construir e iniciar
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

**Nota**: Docker Compose automáticamente fusiona:
- `docker-compose.yml` (configuración base)
- `docker-compose.override.yml` (volúmenes para desarrollo)

Para producción pura (sin volúmenes):
```bash
docker-compose -f docker-compose.yml up -d
```

## Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WhatsApp      │    │   FastAPI       │    │   RAG System    │
│   Business API  │───▶│   Backend       │───▶│   + OpenAI      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Google Cloud    │
                       │ Storage         │
                       │ (Embeddings)    │
                       └─────────────────┘
```

## Procesamiento de Datos

**⚠️ Importante**: Este repositorio contiene solo el backend del chatbot. El procesamiento de documentos PDF se realiza en el repositorio separado `drcecim_upload`.

### Flujo completo:
1. **drcecim_upload**: Procesa PDFs → Genera embeddings → Sube a GCS
2. **chatbot_uba**: Descarga embeddings desde GCS → Responde consultas

## Fallback Local

El sistema incluye una carpeta `data/embeddings/` como fallback:
- Se usa si GCS no está disponible
- Útil para desarrollo local
- Permite testing sin conexión a la nube

## Dependencias Principales

- `fastapi`, `uvicorn` - Servidor web
- `openai` - API de OpenAI
- `faiss-cpu` - Búsquedas vectoriales
- `google-cloud-storage` - Almacenamiento en GCS
- `google-api-python-client` - APIs de Google

## Monitoreo

- **Logs**: Disponibles en `logs/`
- **Health check**: `http://localhost:8080/health`
- **Métricas**: Logs detallados del sistema RAG

## Requisitos

- Python 3.11+
- Cuenta de OpenAI con API key
- Cuenta de WhatsApp Business (opcional)
- Google Cloud Storage (opcional)

---

**Nota**: Para procesamiento de documentos, consulta el repositorio `drcecim_upload`.