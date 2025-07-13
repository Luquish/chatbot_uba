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
├── rag_system.py           # Clase principal del sistema RAG
├── data/
│   ├── raw/                # Documentos PDF sin procesar
│   ├── processed/          # Documentos procesados en formato markdown
│   ├── embeddings/         # Embeddings e índices FAISS generados
│   └── finetuning/         # Datos para fine-tuning del modelo
├── models/                 # Modelos de IA y clases base
│   ├── base_model.py       # Clase base para modelos de LLM
│   ├── openai_model.py     # Implementación para modelos de OpenAI
│   └── finetuned_model/    # Modelos fine-tuneados
├── storage/                # Gestión de almacenamiento vectorial
│   └── vector_store.py     # Implementación de FAISS Vector Store
├── scripts/
│   ├── preprocess.py       # Procesamiento de documentos PDF con Marker
│   ├── create_embeddings.py # Generación de embeddings con OpenAI
│   ├── run_rag.py          # Script de ejecución de consola para el sistema RAG
│   ├── main.py             # Backend FastAPI y webhook de WhatsApp
│   └── train_finetune.py   # Fine-tuning de modelos con OpenAI
├── handlers/               # Manejadores de intenciones y servicios específicos
│   ├── intent_handler.py   # Manejo de intenciones de usuario
│   ├── courses_handler.py  # Manejador de consultas sobre cursos
│   └── calendar_handler.py # Manejador de eventos de calendario
├── services/               # Servicios externos integrados
│   ├── calendar_service.py # Integración con Google Calendar
│   └── sheets_service.py   # Integración con Google Sheets
├── utils/                  # Utilidades del sistema
│   └── date_utils.py       # Utilidades para manejo de fechas
├── config/                 # Archivos de configuración
│   ├── settings.py         # Configuraciones generales del sistema
│   ├── constants.py        # Constantes del sistema
│   └── calendar_config.py  # Configuración de servicios de calendario
├── logs/                   # Archivos de registro
├── docs/                   # Documentación del proyecto
├── Dockerfile              # Configuración para Docker
└── docker-compose.yml      # Configuración de servicios Docker
```

## Configuración

1. Clona el repositorio:
```bash
git clone https://github.com/yourusername/chatbot_uba.git
cd chatbot_uba
```

2. Crea y activa un entorno virtual:
   (Opcional pero recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Crea un archivo `.env` basado en el template `.env.example`:
```bash
cp .env.example .env
```

5. Configura las variables de entorno en el archivo `.env`:

### Variables de Entorno Principales

```
# Configuración de Usuario para Pruebas
MY_PHONE_NUMBER=                                 # (opcional) Tu número para pruebas

# Configuración de Google Calendar API
CALENDAR_ID_EXAMENES=                            # ID del calendario de exámenes
CALENDAR_ID_INSCRIPCIONES=                       # ID del calendario de inscripciones
CALENDAR_ID_CURSADA=                             # ID del calendario de cursada
CALENDAR_ID_TRAMITES=                            # ID del calendario de trámites
GOOGLE_API_KEY=                         # API Key de Google Calendar

# Configuración de WhatsApp Business API
WHATSAPP_API_TOKEN=                              # Token de la API de WhatsApp
WHATSAPP_PHONE_NUMBER_ID=                        # ID del número de teléfono en WhatsApp Business
WHATSAPP_BUSINESS_ACCOUNT_ID=                    # ID de la cuenta de negocio
WHATSAPP_WEBHOOK_VERIFY_TOKEN=                   # Token para verificar webhooks

# URL del backend para recibir mensajes de Glitch
BACKEND_URL=http://localhost:8000/api/whatsapp/message  # URL para webhooks

# Configuración del servidor
HOST=0.0.0.0                                     # Host para el servidor FastAPI
PORT=8000                                        # Puerto para el servidor FastAPI

# Directorios de datos
EMBEDDINGS_DIR=data/embeddings                   # Directorio para embeddings

# Configuración de modelos de OpenAI
OPENAI_API_KEY=                                  # API Key de OpenAI
PRIMARY_MODEL=gpt-4o-mini                        # Modelo principal de OpenAI
FALLBACK_MODEL=gpt-4.1-nano                      # Modelo de respaldo si falla el principal
EMBEDDING_MODEL=text-embedding-3-small           # Modelo para embeddings

# Parámetros de generación
TEMPERATURE=0.7                                  # Temperatura para generación
TOP_P=0.9                                        # Parámetro top_p para generación
TOP_K=50                                         # Parámetro top_k para generación
MAX_OUTPUT_TOKENS=300                            # Máxima longitud de salida

# Configuración de RAG
RAG_NUM_CHUNKS=3                                 # Número de chunks a recuperar
SIMILARITY_THRESHOLD=0.3                         # Umbral de similitud mínima

# Configuración del dispositivo (auto, cuda, cpu, mps)
DEVICE=mps                                       # auto, cuda, cpu, o mps para Mac

# Configuración del entorno
API_TIMEOUT=30                                   # Timeout para llamadas a APIs externas (segundos)
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
uvicorn main:app --reload
```

### Despliegue con Docker

El proyecto está configurado para ser desplegado usando Docker, separando el procesamiento de documentos (local) de la ejecución del servidor (producción).

#### Estructura Docker

```
.
├── Dockerfile           # Configuración de la imagen
├── docker-compose.yml   # Configuración del servicio
└── .dockerignore        # Archivos excluidos del contenedor
```

#### Archivos en Producción

Solo los archivos necesarios para la ejecución se incluyen en el contenedor:
- `main.py`
- `scripts/run_rag.py`
- `scripts/calendar_service.py`
- `scripts/date_utils.py`
- `scripts/gcs_storage.py`
- `scripts/__init__.py`
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

#### Ejecución directa con Docker

Para ejecutar el contenedor Docker directamente con todas las configuraciones necesarias:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data/embeddings:/app/data/embeddings \
  -e PORT=8000 \
  -e OPENAI_API_KEY=your_openai_key \
  -e WHATSAPP_API_TOKEN=your_whatsapp_token \
  -e WHATSAPP_PHONE_NUMBER_ID=your_whatsapp_phone_id \
  -e WHATSAPP_BUSINESS_ACCOUNT_ID=your_whatsapp_business_id \
  -e WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_webhook_token \
  uba-chatbot
```

**Parámetros importantes**:
- Montar embeddings: `-v $(pwd)/data/embeddings:/app/data/embeddings`
- Especificar el puerto: `-e PORT=8000`
- Usar IDs correctos para WhatsApp (el PHONE_NUMBER_ID es un ID numérico asignado por Meta, no el número de teléfono real)

### Flujo de Actualización de Documentos

1. Añadir nuevos PDFs en `data/raw/`
2. Ejecutar localmente el preprocesamiento:
   ```bash
   python scripts/preprocess.py
   python scripts/create_embeddings.py
   ```
3. Los nuevos embeddings se generarán en `data/embeddings/`
4. El contenedor Docker montará automáticamente los nuevos embeddings

## Despliegue en Google Cloud Run

El proyecto está diseñado para ser desplegado en Google Cloud Run, utilizando Google Cloud Storage (GCS) para almacenar los embeddings en producción.

### Configuración inicial en Google Cloud

1. **Autenticar y configurar el proyecto**:
```bash
# Iniciar sesión en Google Cloud
gcloud auth login

# Establecer el proyecto activo
gcloud config set project [PROJECT_ID]

# Activar las APIs necesarias
gcloud services enable artifactregistry.googleapis.com run.googleapis.com storage-api.googleapis.com secretmanager.googleapis.com
```

2. **Crear bucket y subir embeddings**:
```bash
# Crear bucket en la misma región que usará Cloud Run (por ejemplo, us-central1 o southamerica-east1)
gsutil mb -l [REGION] gs://[BUCKET_NAME]

# Subir embeddings usando gcloud storage
gcloud storage cp --recursive data/embeddings/* gs://[BUCKET_NAME]/

# Verificar archivos subidos
gcloud storage ls gs://[BUCKET_NAME]/
```

3. **Crear cuenta de servicio para el chatbot**:
```bash
# Crear una service account específica para el chatbot
gcloud iam service-accounts create chatbot-serviceaccount \
    --display-name="Chatbot Service Account"

# Verificar que se creó correctamente
gcloud iam service-accounts list
```

4. **Crear secretos para credenciales**:
```bash
# OpenAI
echo -n "your_openai_api_key" | gcloud secrets create openai-credentials --data-file=- --replication-policy="automatic"

# WhatsApp (secretos individuales)
echo -n "your_whatsapp_api_token" | gcloud secrets create whatsapp-credentials --data-file=- --replication-policy="automatic"
echo -n "your_phone_number_id" | gcloud secrets create whatsapp-phone-number-id --data-file=- --replication-policy="automatic"
echo -n "your_business_account_id" | gcloud secrets create whatsapp-business-account-id --data-file=- --replication-policy="automatic"
echo -n "your_webhook_verify_token" | gcloud secrets create whatsapp-webhook-verify-token --data-file=- --replication-policy="automatic"
```

5. **Otorgar permisos necesarios**:
```bash
# Usar la cuenta de servicio que creamos
SERVICE_ACCOUNT="chatbot-serviceaccount@[PROJECT_ID].iam.gserviceaccount.com"

# Permitir a Cloud Run acceder a los secretos
gcloud secrets add-iam-policy-binding openai-credentials \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding whatsapp-credentials \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding whatsapp-phone-number-id \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding whatsapp-business-account-id \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding whatsapp-webhook-verify-token \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

# Permitir a Cloud Run acceder al bucket de GCS
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectViewer gs://[BUCKET_NAME]
```

### Adaptaciones para Google Cloud Storage

El código ha sido adaptado para detectar automáticamente si se está ejecutando en un entorno local o en Cloud Run:

- En entorno local: lee los embeddings desde el sistema de archivos local (`data/embeddings/`)
- En Cloud Run: lee los embeddings desde el bucket de GCS configurado en la variable de entorno `GCS_BUCKET_NAME`

Esta adaptación está implementada en el módulo `scripts/gcs_storage.py`, que proporciona funciones para:
- Listar archivos en un bucket de GCS
- Descargar archivos de GCS a una ubicación local temporal
- Leer archivos como strings o datos binarios directamente desde GCS
- Cargar índices FAISS y sus metadatos desde GCS

### Despliegue paso a paso en Google Cloud Run

1. **Verificar el archivo Dockerfile**:
   Asegúrate de que el Dockerfile incluya todos los archivos necesarios, especialmente `scripts/__init__.py`:
   ```dockerfile
   # Copiar solo los archivos necesarios para producción
   COPY requirements-prod.txt .
   COPY main.py .
   COPY scripts/run_rag.py scripts/
   COPY scripts/calendar_service.py scripts/
   COPY scripts/date_utils.py scripts/
   COPY scripts/gcs_storage.py scripts/
   COPY scripts/__init__.py scripts/
   COPY config/calendar_config.py config/
   ```

2. **Construir imagen Docker para arquitectura amd64** (necesario para Cloud Run):
   ```bash
   docker buildx build --platform linux/amd64 -t uba-chatbot:latest .
   docker tag uba-chatbot:latest gcr.io/[PROJECT_ID]/uba-chatbot:latest
   ```

3. **Autenticar Docker con Google Cloud y subir imagen**:
   ```bash
   gcloud auth configure-docker
   docker push gcr.io/[PROJECT_ID]/uba-chatbot:latest
   ```

4. **Crear archivo cloud-run.yaml** basado en el ejemplo:
   ```bash
   cp cloud-run.yaml.example cloud-run.yaml
   # Editar cloud-run.yaml para reemplazar los placeholders con tus valores reales
   ```

5. **Configurar anotaciones importantes en cloud-run.yaml**:
   ```yaml
   run.googleapis.com/startup-probe-failure-threshold: "5"
   ```
   Esta anotación es crucial para dar más tiempo al contenedor para inicializarse, especialmente cuando carga embeddings desde GCS.

6. **Desplegar en Cloud Run**:
   ```bash
   gcloud run services replace cloud-run.yaml --region=[REGION]
   ```

7. **Permitir acceso público** (necesario para webhooks de WhatsApp):
   ```bash
   gcloud run services add-iam-policy-binding [SERVICE_NAME] \
     --member="allUsers" \
     --role="roles/run.invoker" \
     --region=[REGION]
   ```

8. **Verificar el despliegue**:
   ```bash
   # Obtener URL del servicio
   gcloud run services describe [SERVICE_NAME] --region=[REGION] --format="value(status.url)"
   
   # Verificar el estado del servicio
   curl -v [SERVICE_URL]/health
   
   # Probar el webhook de WhatsApp (con el token real)
   curl -v "[SERVICE_URL]/webhook/whatsapp?hub.mode=subscribe&hub.challenge=12345&hub.verify_token=[YOUR_VERIFY_TOKEN]"
   
   # Enviar un mensaje de prueba
   curl -v [SERVICE_URL]/test-webhook
   ```

9. **Configurar en WhatsApp Business**:
   1. Ve al [Facebook Developer Portal](https://developers.facebook.com/)
   2. Selecciona tu app de WhatsApp Business
   3. Navega a "WhatsApp" > "API Setup" > "Configuration"
   4. En "Webhook", configura:
      - **Callback URL**: `[SERVICE_URL]/webhook/whatsapp`
      - **Verify Token**: El token configurado en `WHATSAPP_WEBHOOK_VERIFY_TOKEN`
   5. Selecciona los eventos: `messages`, `message_deliveries`, `message_reads`
   6. Haz clic en "Verify and Save"

### Consideraciones importantes para el despliegue

1. **Tiempo de inicio**: El contenedor necesita tiempo para inicializarse correctamente, especialmente al cargar embeddings desde GCS. La anotación `startup-probe-failure-threshold: "5"` ayuda a extender este tiempo.

2. **Puerto de escucha**: Cloud Run proporciona automáticamente una variable de entorno `PORT` que debe ser utilizada por la aplicación. No intentes establecer esta variable manualmente en `cloud-run.yaml`, ya que es reservada del sistema.

3. **Errores comunes**:
   - Si el script no encuentra los embeddings, verifica que el bucket GCS esté correctamente configurado y que los archivos estén presentes.
   - Si la carga desde GCS falla, asegúrate de que `scripts/__init__.py` esté incluido en el Dockerfile para permitir importaciones correctas.

4. **Monitoreo de logs**:
   ```bash
   # Ver logs en tiempo real
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=[SERVICE_NAME]" --limit=50
   ```

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