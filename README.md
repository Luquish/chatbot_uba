# Chatbot Educativo UBA - Facultad de Medicina

Este proyecto implementa un chatbot educativo para la Facultad de Medicina de la Universidad de Buenos Aires (UBA), diseñado para asistir a aproximadamente 127.000 estudiantes con consultas administrativas.

## Características Principales

- **Modelo Base**: Utiliza Mistral 7B o Gemma 7B como modelo base
- **Fine-tuning**: Implementa LoRA/QLoRA para adaptación específica
- **RAG**: Sistema de recuperación aumentada de información
- **Integración WhatsApp**: Compatible con Twilio y API oficial de WhatsApp
- **Backend**: API REST con FastAPI
- **Almacenamiento Dual**: FAISS local para desarrollo, Pinecone para producción

## Guía Paso a Paso de Implementación

Sigue estos pasos en orden para configurar completamente el proyecto:

### 1. Configuración del Entorno

```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Unix/macOS
# o
.\venv\Scripts\activate  # En Windows
# o con conda
conda create -n chatbots_uba python=3.9 -y
conda activate chatbots_uba

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### 2. Preparación de Datos

```bash
# Colocar tus documentos PDF en la carpeta data/raw
mkdir -p data/raw
# [Agrega aquí tus PDFs]

# Preprocesar documentos
python scripts/preprocess.py
```

Este paso:
- Extrae texto de documentos PDF
- Limpia y normaliza el texto
- Divide el texto en fragmentos (chunks) manejables
- Guarda los resultados en `data/processed/`

### 3. Generación de Embeddings

```bash
# Crear embeddings para los documentos procesados
python scripts/create_embeddings.py
```

Este paso:
- Genera embeddings vectoriales para cada fragmento de texto
- En modo desarrollo: almacena los embeddings en un índice FAISS local
- En modo producción: almacena los embeddings en Pinecone
- Los índices locales se guardan en `data/embeddings/`

### 4. Fine-tuning del Modelo (Opcional)

```bash
# Fine-tunear el modelo base para adaptarlo al dominio
python scripts/train_finetune.py
```

Este paso:
- Aplica técnicas de LoRA/QLoRA para adaptar el modelo base
- El modelo resultante se guarda en `models/finetuned_model/`
- Puede tardar varias horas dependiendo del hardware disponible

### 5. Configuración de WhatsApp

#### Para desarrollo (Twilio):

1. Crear una cuenta en [Twilio](https://www.twilio.com/)
2. Activar el sandbox de WhatsApp en Twilio
3. En el panel de Twilio, busca la sección "WhatsApp Sandbox"
4. Envía el mensaje de unión (ejemplo: "join chicken-firm") desde tu WhatsApp al número de Twilio
5. Configura las variables de Twilio en `.env`:
   ```
   ENVIRONMENT=development
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_PHONE_NUMBER=your_phone_number
   MY_PHONE_NUMBER=your_whatsapp_number
   ```

#### Para producción (API oficial):

1. Configura la API oficial de WhatsApp Business en `.env`:
   ```
   ENVIRONMENT=production
   WHATSAPP_API_TOKEN=your_api_token
   WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
   WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id
   ```

### 6. Iniciar el Servidor

```bash
# Iniciar el servidor de desarrollo
python scripts/deploy_backend.py
```

### 7. Configurar Webhook con ngrok

```bash
# Instalar ngrok si no lo tienes
# Exponer el servidor local a internet
ngrok http 8000
```

Después:
1. Copia la URL generada por ngrok (ej: https://abc123.ngrok.io)
2. En el panel de Twilio, en la sección "WhatsApp Sandbox"
3. Configura la URL de webhook como:
   ```
   https://tu-url-ngrok.io/webhook/whatsapp
   ```
4. Selecciona el método HTTP como "POST"
5. Guarda los cambios

### 8. Probar el Chatbot

```bash
# Enviar un mensaje de prueba
# Visita en tu navegador
http://localhost:8000/test-message
```

o envía un mensaje desde tu WhatsApp al número de Twilio.

## Sistema RAG y Prompt del LLM

El proyecto utiliza un sistema RAG (Retrieval-Augmented Generation) que:

1. Recibe una consulta del usuario
2. Recupera los fragmentos más relevantes de la información almacenada
3. Construye un prompt con el contexto recuperado
4. Genera una respuesta usando el modelo fine-tuneado

### Prompt Utilizado

El prompt usado para generar respuestas tiene esta estructura:

```
Contexto:
[Aquí se insertan los fragmentos de documentos relevantes]

Pregunta: [Consulta del usuario]

Respuesta:
```

Este prompt se encuentra en el archivo `scripts/run_rag.py`, método `generate_response()`.

## Estructura del Proyecto

```
project/
├── data/
│   ├── raw/                # PDFs y documentos sin procesar
│   ├── processed/          # Datos preprocesados
│   ├── embeddings/         # Índices de embeddings (FAISS local)
│   └── finetuning/         # Dataset para fine-tuning
├── models/
│   ├── base_model/         # Modelo base
│   └── finetuned_model/    # Modelo fine-tuneado
├── scripts/
│   ├── preprocess.py       # Preprocesamiento de documentos
│   ├── create_embeddings.py# Generación de embeddings
│   ├── train_finetune.py   # Fine-tuning del modelo
│   ├── run_rag.py          # Sistema RAG
│   └── deploy_backend.py   # Backend y API
└── notebooks/              # Experimentación y análisis
```

## Requisitos

- Python 3.9+
- CUDA compatible GPU (para fine-tuning o producción con NVIDIA)
- 16GB+ RAM
- Para desarrollo en Mac:
  - Apple Silicon (M1/M2/M3/M4)
  - MLX y mlx-lm (optimización para Apple Silicon)
- Para producción:
  - bitsandbytes (cuantización para servidores)
  - torch >= 2.0
- Dependencias principales (ver requirements.txt para la lista completa):
  ```
  transformers>=4.36.0
  torch>=2.0.0
  sentence-transformers>=2.2.2
  faiss-cpu>=1.7.4  # o faiss-gpu
  pinecone-client>=2.2.2  # para producción
  fastapi>=0.104.1
  uvicorn>=0.24.0
  python-dotenv>=1.0.0
  ```

## Configuración de Base de Datos Vectorial

El sistema está diseñado para funcionar con diferentes bases de datos vectoriales según el entorno:

### 1. Desarrollo (FAISS local)

- En desarrollo, los embeddings se almacenan localmente usando FAISS
- Ubicación: `data/embeddings/faiss_index.bin`
- Adecuado para pruebas y desarrollo
- No requiere servicios externos

### 2. Producción (Pinecone)

- En producción, los embeddings se almacenan en Pinecone (base de datos vectorial en la nube)
- Ventajas:
  - Alta disponibilidad
  - Escalabilidad
  - Búsqueda vectorial optimizada
  - Sin necesidad de almacenamiento local

Para configurar Pinecone:

1. Crear una cuenta en [Pinecone](https://www.pinecone.io/)
2. Crear un índice coseno con la dimensión adecuada (384 para el modelo de embedding predeterminado)
3. Configurar las variables en `.env`:
   ```
   ENVIRONMENT=production  # Activa el uso de Pinecone
   PINECONE_API_KEY=your_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=uba-chatbot-embeddings
   ```

### 3. Migración de datos

Para transferir embeddings de desarrollo a producción:
1. Configure el entorno de producción en `.env`
2. Ejecute nuevamente `python scripts/create_embeddings.py`
3. Sus embeddings se cargarán automáticamente en Pinecone

## Configuración de WhatsApp

El sistema está diseñado para funcionar con dos proveedores de WhatsApp, seleccionados automáticamente según el entorno:

### 1. Configuración de entorno

En tu archivo `.env`, establece el entorno:
```
# Opciones: development, production
ENVIRONMENT=development  # Usa Twilio para desarrollo
# o
ENVIRONMENT=production   # Usa API oficial de WhatsApp para producción
```

### 2. Desarrollo con Twilio

Para desarrollo local con Twilio:

1. Crear una cuenta en [Twilio](https://www.twilio.com/)
2. Activar el sandbox de WhatsApp en Twilio
3. Configurar las variables en `.env`:
   ```
   ENVIRONMENT=development
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_PHONE_NUMBER=your_phone_number
   MY_PHONE_NUMBER=your_personal_whatsapp_number  # Con código de país
   ```
4. **Importante: Unirse al sandbox de Twilio**:
   - En la consola de Twilio, busca la sección "WhatsApp Sandbox"
   - Verás un código de unión (ejemplo: "join chicken-firm")
   - Desde tu WhatsApp (el número configurado como MY_PHONE_NUMBER), envía este mensaje exacto al número de WhatsApp de Twilio
   - Recibirás un mensaje de confirmación cuando estés conectado
   - **Solo los números que se han unido al sandbox pueden recibir mensajes**

5. Iniciar el servidor:
   ```bash
   python scripts/deploy_backend.py
   ```
6. Probar la conexión:
   ```
   # Abrir en navegador
   http://localhost:8000/test-message
   ```
   Deberías recibir un mensaje en tu WhatsApp si todo está configurado correctamente.

7. Para recibir mensajes externos, exponer el servidor:
   ```bash
   ngrok http 8000
   ```
8. Configurar el webhook en Twilio apuntando a:
   ```
   https://tu-url-ngrok.io/webhook/whatsapp
   ```

### 3. Producción con API oficial de WhatsApp

Para entorno de producción:

1. Crear una cuenta en [Facebook Business Manager](https://business.facebook.com/)
2. Solicitar acceso a la API de WhatsApp Business
3. Crear un número de teléfono en WhatsApp Business y obtener credenciales
4. Configurar las variables en `.env`:
   ```
   ENVIRONMENT=production
   WHATSAPP_API_TOKEN=your_api_token
   WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
   WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id
   ```
5. Desplegar el servidor en un entorno de producción
6. Configurar el webhook de Meta apuntando a:
   ```
   https://tu-dominio.com/webhook/whatsapp
   ```

## Optimización de Modelos

El sistema implementa una estrategia dual de optimización para maximizar el rendimiento en diferentes entornos:

### 1. Desarrollo en Mac (Apple Silicon)

Si estás usando un dispositivo Mac con chip Apple Silicon (M1, M2, M3, M4):

- **MLX**: Framework de Apple optimizado para Apple Silicon
- **Beneficios**:
  - Rendimiento hasta 10x más rápido que PyTorch en Apple Silicon
  - Menor consumo de memoria
  - Aprovecha los Neural Engine específicos del chip
  - Sin necesidad de cuantización manual

Para activar MLX:
```
# En .env
USE_MLX=True
USE_8BIT=False
USE_4BIT=False
```

Esta configuración automáticamente:
- Detecta modelos Mistral y los carga usando MLX
- Busca versiones optimizadas para MLX de otros modelos
- Funciona sin cambios de código entre Mistral y otros modelos

Requisitos:
```bash
pip install mlx mlx-lm
```

### 2. Producción en Servidores Cloud

Para despliegue en servidores de producción (típicamente Linux con GPUs NVIDIA):

- **bitsandbytes**: Biblioteca de cuantización de alta performance
- **Beneficios**:
  - Cuantización INT8 (8-bit): Reduce memoria en ~50% con mínima pérdida de calidad
  - Cuantización INT4 (4-bit): Reduce memoria en ~75% con pérdida moderada
  - Compatible con GPUs NVIDIA en servidores cloud
  - Mayor throughput y capacidad de usuarios concurrentes

Para activar bitsandbytes:
```
# En .env
USE_MLX=False
USE_8BIT=True   # Para cuantización de 8 bits
# o
USE_4BIT=True   # Para cuantización de 4 bits (más agresiva)
```

### Detección Automática

El sistema detecta automáticamente:
1. Si está en Apple Silicon (Mac) o en un servidor standard
2. Si MLX está disponible
3. El tipo de modelo que está cargando

Y aplica la optimización adecuada sin cambios manuales. Esta estrategia dual permite:
- Desarrollo eficiente y rápido en MacBooks con M1/M2/M3/M4
- Despliegue económico y escalable en servidores de producción

## Variables de entorno completas

Crea un archivo `.env` con las siguientes variables:

```
# Configuración de entorno
ENVIRONMENT=development  # Opciones: development, production

# Configuración de Usuario para Pruebas
MY_PHONE_NUMBER=your_personal_whatsapp_number

# Configuración de Twilio (para desarrollo)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_phone_number

# Configuración de WhatsApp Business API oficial (para producción)
WHATSAPP_API_TOKEN=your_api_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id

# Configuración de almacenamiento vectorial (Pinecone)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=uba-chatbot-embeddings

# Configuración del servidor
HOST=0.0.0.0
PORT=8000

# Configuración del modelo
MODEL_PATH=models/finetuned_model
EMBEDDINGS_DIR=data/embeddings
BASE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
FALLBACK_MODEL_NAME=google/gemma-2b

# Configuración de Hugging Face
HUGGING_FACE_HUB_TOKEN=your_huggingface_token

# Configuración de optimización
USE_MLX=True    # Habilitar MLX para Apple Silicon
USE_8BIT=False  # Cuantización de 8-bit (para servidores)
USE_4BIT=False  # Cuantización de 4-bit (para servidores)
```

## Contribución

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Tu Nombre - [@tutwitter](https://twitter.com/tutwitter) - email@example.com

Link del Proyecto: [https://github.com/tu-usuario/chatbot_uba](https://github.com/tu-usuario/chatbot_uba)