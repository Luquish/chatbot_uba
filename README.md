# Chatbot Administrativo UBA

Un sistema de chatbot administrativo para la Universidad de Buenos Aires (UBA) basado en tecnología RAG (Retrieval Augmented Generation) e integración con Telegram. El sistema proporciona respuestas precisas y contextuales a consultas administrativas utilizando documentos institucionales.

## Descripción General

Este proyecto implementa un asistente virtual para cualquier facultad de la UBA que responde consultas administrativas a través de Telegram. El sistema utiliza técnicas avanzadas de procesamiento de lenguaje natural:

### Características Principales

- **Sistema RAG**: Generación Aumentada por Recuperación que combina recuperación de información y generación de respuestas contextuales.
- **Integración Telegram**: Conexión directa con la API de Telegram Bot para enviar y recibir mensajes.
- **Sistema de Intenciones**: Router declarativo por YAML + herramientas (tools). El LLM elige tool (conversational, calendario, sheets, FAQs, RAG) y hay reglas mínimas como respaldo.
- **Almacenamiento Híbrido**: Carga embeddings desde Google Cloud Storage con fallback local.
- **Integración con Google APIs**: Calendario y Sheets para información dinámica.

### Sesiones y Contexto Conversacional
- Sesiones en memoria con TTL configurable (30 min por defecto).
- Soporta consultas relativas: “y la que sigue?” usa el contexto previo (semana/mes) para responder.
- `SessionService` limpia sesiones expiradas automáticamente y guarda metadatos útiles (tipo de consulta, intención de calendario, mes consultado, user_name).

#### Configuración de sesiones efímeras
- `SESSION_TTL_SECONDS`: tiempo de vida de la sesión en segundos (default: 1800 = 30 min)
- `SESSION_SWEEPER_INTERVAL`: intervalo del limpiador en segundos (default: 60)
- El limpiador corre en segundo plano y se detiene automáticamente en el evento de apagado del servidor.

### Métricas y Enrutamiento
- Logs de router: tool ejecutada, score y fallback si ninguna produce respuesta.
- YAML (`config/router.yaml`) para ajustar prioridades, triggers y umbrales sin tocar código.

### Flujo del Sistema

1. El estudiante envía una consulta por Telegram
2. El backend de FastAPI recibe el mensaje a través de un webhook
3. El sistema RAG analiza la consulta e identifica la intención
4. Se recupera información relevante desde la base de conocimientos
5. Se genera una respuesta utilizando la información contextual y la consulta
6. La respuesta se envía de vuelta al estudiante a través de Telegram

## Estructura del Proyecto

```
chatbot_uba/                     # Backend del chatbot
├── main.py                      # FastAPI + Telegram webhook
├── rag_system.py               # Sistema RAG principal
├── requirements.txt            # Dependencias (dev + prod unificadas)
├── Dockerfile                  # Imagen de contenedor
├── docker-compose.yml          # Configuración base
├── docker-compose.override.yml # Volúmenes para desarrollo
├── data/
│   └── embeddings/            # Fallback local para embeddings
├── config/                    # Configuraciones del sistema
├── models/                    # Modelos OpenAI
├── storage/                   # Vector store (PostgreSQL/pgvector)
├── handlers/                  # Manejadores de intenciones
├── services/                  # Servicios externos (Calendar, Sheets)
├── utils/                     # Utilidades del sistema
└── scripts/
    ├── setup_database.py     # Inicializa tablas e índices en PostgreSQL
    └── logs/                 # Logs de scripts
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

# Telegram Bot API
TELEGRAM_BOT_TOKEN=your-bot-token-from-botfather
TELEGRAM_WEBHOOK_SECRET=your-webhook-secret
TELEGRAM_ADMIN_USER_ID=your-user-id

# Google Cloud Storage (opcional)
USE_GCS=true
GCS_BUCKET_NAME=your-bucket-name
GCS_CREDENTIALS_PATH=./cloud_functions/credentials/service-account.json

# Google APIs (opcional)
GOOGLE_API_KEY=your-google-api-key
CURSOS_SPREADSHEET_ID=your-spreadsheet-id

# Configuración del servidor
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8080

# Sesiones efímeras (opcional)
SESSION_TTL_SECONDS=1800
SESSION_SWEEPER_INTERVAL=60

# Métricas (opcional)
METRICS_API_KEY=coloca-un-token-seguro
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

### Endpoints útiles y métricas

- `GET /health`: estado del servicio. Incluye `session_stats` (sesiones activas, TTL) y `router_metrics`.
- `GET /metrics`: requiere header `X-API-Key` con `METRICS_API_KEY` definido en `.env`.
- Webhook de Telegram: `POST /webhook/telegram`.

Apagado limpio: el servicio detiene el limpiador de sesiones para evitar hilos colgando.

## Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram      │    │   FastAPI       │    │   RAG System    │
│   Bot API       │───▶│   Backend       │───▶│   + OpenAI      │
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
- `pgvector` - Búsquedas vectoriales en PostgreSQL
- `cloud-sql-python-connector` - Conexión a Cloud SQL
- `sqlalchemy` - ORM para PostgreSQL
- `google-cloud-storage` - Almacenamiento en GCS
- `google-api-python-client` - APIs de Google

## Monitoreo

- **Logs**: Disponibles en `logs/`
- **Health check**: `http://localhost:8080/health`
- **Métricas**: usar `GET /metrics` con header `X-API-Key: $METRICS_API_KEY`

## Testing rápido

- Sesiones (TTL y relativas):
```bash
python tests/test_sessions.py
```

- Suite completa (requiere servicios externos configurados):
```bash
python tests/run_tests.py
```

## Requisitos

- Python 3.11+
- Cuenta de OpenAI con API key
- Bot de Telegram (crear con @BotFather)
- Google Cloud Storage (opcional)

---

**Nota**: Para procesamiento de documentos, consulta el repositorio `drcecim_upload`.