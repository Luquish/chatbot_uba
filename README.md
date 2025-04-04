# Chatbot UBA Medicina

Chatbot educativo para la Facultad de Medicina de la UBA, diseñado para asistir a aproximadamente 127,000 estudiantes con consultas administrativas.

## Estructura del Proyecto

El proyecto está organizado en los siguientes scripts principales:

1. **Preprocesamiento de Datos** (`scripts/preprocess.py`):
   - Extrae texto de documentos PDF
   - Limpia y segmenta el texto
   - Prepara los datos para la generación de embeddings

2. **Generación de Embeddings** (`scripts/create_embeddings.py`):
   - Crea embeddings vectoriales del texto procesado
   - Almacena los embeddings en FAISS para desarrollo
   - Utiliza Pinecone para producción

3. **Fine-tuning del Modelo** (`scripts/train_finetune.py`):
   - Adapta el modelo base usando técnicas LoRA/QLoRA
   - Optimiza el modelo para el dominio específico

4. **Sistema RAG** (`scripts/run_rag.py`):
   - Maneja consultas de usuarios
   - Recupera información relevante
   - Genera respuestas contextuales

5. **Backend y API** (`scripts/deploy_backend.py`):
   - Implementa el backend con FastAPI
   - Gestiona la integración con WhatsApp Business API

## Configuración

### Variables de Entorno

Crear un archivo `.env` con las siguientes variables:

```env
# Entorno
ENVIRONMENT=development  # o production

# WhatsApp Business API
WHATSAPP_API_TOKEN=your_api_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_webhook_verify_token

# Número de teléfono para pruebas
MY_PHONE_NUMBER=your_test_phone_number

# Modelo y embeddings
MODEL_PATH=models/finetuned_model
EMBEDDINGS_DIR=data/embeddings

# Servidor
HOST=0.0.0.0
PORT=8000
```

### Configuración de WhatsApp

1. Crear una cuenta de negocio en [Meta Business Manager](https://business.facebook.com/)
2. Configurar la API de WhatsApp Business:
   - Obtener el token de acceso
   - Registrar el número de teléfono
   - Obtener el ID del número de teléfono
   - Obtener el ID de la cuenta de negocio

3. Configurar el webhook:
   - URL: `https://tu-dominio.com/webhook/whatsapp`
   - Token de verificación: usar el valor de `WHATSAPP_WEBHOOK_VERIFY_TOKEN`
   - Eventos a suscribir:
     - messages
     - message_template_status_updates

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/chatbot_uba.git
cd chatbot_uba
```

2. Crear y activar un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Desarrollo

1. Iniciar el servidor de desarrollo:
```bash
python scripts/deploy_backend.py
```

2. Probar el chatbot:
   - Usar el endpoint `/chat` para pruebas sin WhatsApp
   - Usar el endpoint `/test-message` para enviar mensajes de prueba
   - Verificar el estado del servicio con `/health`

### Producción

1. Configurar el entorno de producción:
   - Actualizar las variables de entorno
   - Configurar el servidor web (Nginx, Apache, etc.)
   - Configurar SSL/TLS

2. Iniciar el servidor:
```bash
ENVIRONMENT=production python scripts/deploy_backend.py
```

## Seguridad

- Validación HMAC SHA256 para webhooks
- Manejo seguro de tokens
- Logging detallado para monitoreo
- Validación de solicitudes

## Contribución

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.