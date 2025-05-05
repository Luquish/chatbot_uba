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
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectViewer gs://uba-chatbot-embeddings
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

### Construcción de imagen multiplataforma para Cloud Run

Cloud Run ejecuta contenedores en arquitectura x86_64 (amd64). Si estás desarrollando en una máquina con arquitectura diferente (como Apple Silicon/ARM), debes construir específicamente para amd64:

```bash
# Construir imagen específica para amd64 con Docker BuildX
docker buildx build --platform linux/amd64 -t uba-chatbot:latest .

# Etiquetar la imagen para Google Container Registry
docker tag uba-chatbot:latest gcr.io/[PROJECT_ID]/uba-chatbot:amd64

# Subir la imagen a Google Container Registry
docker push gcr.io/[PROJECT_ID]/uba-chatbot:amd64
```

### Configuración de Cloud Run (cloud-run.yaml)

Crea un archivo `cloud-run.yaml` en la raíz del proyecto con la siguiente configuración:

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: [SERVICE_NAME]
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/execution-environment: gen2
        run.googleapis.com/cpu-throttling: "true"
        run.googleapis.com/startup-cpu-boost: "true"
    spec:
      serviceAccountName: [SERVICE_ACCOUNT_NAME]@[PROJECT_ID].iam.gserviceaccount.com
      containers:
      - image: gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: HOST
          value: "0.0.0.0"
        - name: GCS_BUCKET_NAME
          value: "[BUCKET_NAME]"
        - name: MY_PHONE_NUMBER
          value: "+[PHONE_NUMBER]"  # Número para pruebas (opcional)
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-credentials
              key: latest
        - name: WHATSAPP_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: whatsapp-credentials
              key: latest
        - name: WHATSAPP_PHONE_NUMBER_ID
          valueFrom:
            secretKeyRef:
              name: whatsapp-phone-number-id
              key: latest
        - name: WHATSAPP_BUSINESS_ACCOUNT_ID
          valueFrom:
            secretKeyRef:
              name: whatsapp-business-account-id
              key: latest
        - name: WHATSAPP_WEBHOOK_VERIFY_TOKEN
          valueFrom:
            secretKeyRef:
              name: whatsapp-webhook-verify-token
              key: latest
        resources:
          limits:
            cpu: "1"
            memory: "2Gi"
```

### Despliegue en Cloud Run

#### Método 1: Despliegue con archivo YAML
```bash
# Desplegar en Cloud Run usando el archivo YAML
gcloud run services replace cloud-run.yaml --region=[REGION]
```

#### Método 2: Despliegue con comandos
```bash
# Desplegar en Cloud Run directamente con comando
gcloud run deploy [SERVICE_NAME] \
  --image=gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64 \
  --platform=managed \
  --region=[REGION] \
  --allow-unauthenticated \
  --service-account=[SERVICE_ACCOUNT_NAME]@[PROJECT_ID].iam.gserviceaccount.com \
  --set-env-vars="ENVIRONMENT=production,HOST=0.0.0.0,GCS_BUCKET_NAME=[BUCKET_NAME],MY_PHONE_NUMBER=+[PHONE_NUMBER]" \
  --set-secrets="OPENAI_API_KEY=openai-credentials:latest,WHATSAPP_API_TOKEN=whatsapp-credentials:latest,WHATSAPP_PHONE_NUMBER_ID=whatsapp-phone-number-id:latest,WHATSAPP_BUSINESS_ACCOUNT_ID=whatsapp-business-account-id:latest,WHATSAPP_WEBHOOK_VERIFY_TOKEN=whatsapp-webhook-verify-token:latest"
```

### Configuración de acceso público (fundamental para webhooks)

Para que WhatsApp pueda enviar eventos al webhook, el servicio debe ser accesible públicamente:

```bash
# Permitir invocaciones no autenticadas (necesario para webhooks)
gcloud run services add-iam-policy-binding [SERVICE_NAME] \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --region=[REGION]
```

### Verificación del despliegue

Después de desplegar, verifica que todo funcione correctamente:

```bash
# Obtener URL del servicio
gcloud run services describe [SERVICE_NAME] --region=[REGION] --format="value(status.url)"

# Verificar el estado del servicio
curl -v $(gcloud run services describe [SERVICE_NAME] --region=[REGION] --format="value(status.url)")/health

# Verificar el webhook de WhatsApp
curl -v $(gcloud run services describe [SERVICE_NAME] --region=[REGION] --format="value(status.url)")/webhook/whatsapp

# Probar el envío de mensajes
curl $(gcloud run services describe [SERVICE_NAME] --region=[REGION] --format="value(status.url)")/test-webhook
```

### Solución de problemas comunes

#### Problema 1: Error de arquitectura
Si recibes el error "exec format error" o "no matching manifest for linux/amd64", significa que intentas ejecutar una imagen construida para una arquitectura distinta.
```bash
# Solución: Construir específicamente para amd64
docker buildx build --platform linux/amd64 -t [IMAGE_NAME]:latest .
docker tag [IMAGE_NAME]:latest gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64-fix
docker push gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64-fix
gcloud run deploy [SERVICE_NAME] --image gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64-fix --platform managed --region=[REGION] --allow-unauthenticated
```

#### Problema 2: Error al cargar embeddings desde GCS
Si recibes errores relacionados con la carga de embeddings desde GCS, verifica:

1. **Comprobar existencia y estructura del bucket**:
```bash
gcloud storage ls gs://[BUCKET_NAME]/
```

2. **Verificar que las variables de entorno estén configuradas**:
```bash
gcloud run services describe [SERVICE_NAME] --region=[REGION]
```

3. **Solución común**: El error `FAISSVectorStore object has no attribute 'similarity_threshold'` ocurre cuando la carga desde GCS es exitosa pero no se inicializa el atributo. Asegúrate de que este se define al inicio del método `__init__` de la clase `FAISSVectorStore`.

#### Problema 3: Error 403 Forbidden
Si recibes un error 403 al intentar acceder al servicio, verifica los permisos de IAM:

```bash
# Verificar la política actual
gcloud run services get-iam-policy [SERVICE_NAME] --region=[REGION]

# Solución: Configurar acceso público explícitamente
gcloud run services add-iam-policy-binding [SERVICE_NAME] \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --region=[REGION]
```

#### Problema 4: Error con Variable PORT
Cloud Run asigna automáticamente una variable de entorno `PORT`. El código debe usar esta variable en lugar de definir un puerto fijo:

```python
# En scripts/deploy_backend.py
port = int(os.getenv("PORT", "8080"))
```

### Configuración de webhook en Meta para WhatsApp Business

1. Ve al [Facebook Developer Portal](https://developers.facebook.com/) y selecciona tu app
2. Navega a "WhatsApp" > "API Setup" > "Configuration"
3. En "Webhook", configura:
   - **Callback URL**: `https://[TU-SERVICIO-URL]/webhook/whatsapp`
   - **Verify Token**: El mismo valor que configuraste en el secreto `WHATSAPP_WEBHOOK_VERIFY_TOKEN`
4. Selecciona los eventos a suscribir: `messages`, `message_deliveries`, `message_reads`
5. Haz clic en "Verify and Save"

### Monitoreo y logs en Cloud Run

```bash
# Ver logs en tiempo real
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=[SERVICE_NAME]" --limit=50

# Filtrar logs de errores
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=[SERVICE_NAME] AND severity>=ERROR" --limit=10

# Ver logs específicos de una revisión
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=[SERVICE_NAME] AND resource.labels.revision_name=[REVISION-ID]" --limit=20
```

### Actualización del servicio

Para actualizar el servicio después de cambios en el código o configuración:

```bash
# Reconstruir la imagen con nueva versión
docker buildx build --platform linux/amd64 -t [IMAGE_NAME]:latest .
docker tag [IMAGE_NAME]:latest gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64-v2
docker push gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64-v2

# Actualizar el servicio con la nueva imagen
gcloud run services update [SERVICE_NAME] \
  --image gcr.io/[PROJECT_ID]/[IMAGE_NAME]:amd64-v2 \
  --region=[REGION]
```

### Actualización de variables de entorno

Para actualizar variables de entorno sin reconstruir la imagen:

```bash
gcloud run services update [SERVICE_NAME] \
  --region=[REGION] \
  --set-env-vars="NUEVA_VARIABLE=valor,OTRA_VARIABLE=otro_valor"
```

## Endpoints Disponibles

- `GET /health`: Estado del servicio
- `POST /chat`: Endpoint para consultas directas
- `POST /webhook/whatsapp`: Webhook para mensajes de WhatsApp
- `GET /test-webhook`: Envía mensaje de prueba

## Monitoreo y Mantenimiento

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