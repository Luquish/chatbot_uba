import os
import logging
import json
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Any
from enum import Enum
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
from run_rag import RAGSystem
import uvicorn

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determinar el entorno (desarrollo o producción)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
logger.info(f"Iniciando en entorno: {ENVIRONMENT}")

# Configuración de WhatsApp
class WhatsAppProvider(str, Enum):
    TWILIO = "twilio"
    OFFICIAL_API = "official_api"

# Determinar el proveedor basado en el entorno
WHATSAPP_PROVIDER = WhatsAppProvider.TWILIO if ENVIRONMENT == 'development' else WhatsAppProvider.OFFICIAL_API
logger.info(f"Usando proveedor de WhatsApp: {WHATSAPP_PROVIDER}")

# Configuración de Twilio
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Configuración de WhatsApp Business API oficial
WHATSAPP_API_TOKEN = os.getenv('WHATSAPP_API_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID')

# Número de teléfono para pruebas
MY_PHONE_NUMBER = os.getenv('MY_PHONE_NUMBER')

# Inicializar clientes de WhatsApp según el proveedor
twilio_client = None
twilio_validator = None

if WHATSAPP_PROVIDER == WhatsAppProvider.TWILIO and all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)
    logger.info("Cliente Twilio inicializado correctamente")
elif WHATSAPP_PROVIDER == WhatsAppProvider.OFFICIAL_API and all([WHATSAPP_API_TOKEN, WHATSAPP_PHONE_NUMBER_ID]):
    logger.info("Configuración de WhatsApp API oficial inicializada correctamente")
else:
    logger.warning(f"Falta configuración para el proveedor {WHATSAPP_PROVIDER}. La integración con WhatsApp no estará completamente disponible.")

# Clase abstracta para manejar envío de mensajes de WhatsApp
class WhatsAppHandler:
    async def validate_request(self, request: Request) -> bool:
        raise NotImplementedError("Este método debe ser implementado por las subclases")
    
    async def parse_message(self, request: Request) -> Dict[str, str]:
        raise NotImplementedError("Este método debe ser implementado por las subclases")
    
    async def send_message(self, to: str, body: str) -> Dict[str, Any]:
        raise NotImplementedError("Este método debe ser implementado por las subclases")

# Implementación para Twilio
class TwilioWhatsAppHandler(WhatsAppHandler):
    def __init__(self, client: Client, validator: RequestValidator, phone_number: str):
        self.client = client
        self.validator = validator
        self.phone_number = phone_number
    
    async def validate_request(self, request: Request) -> bool:
        if not all([self.client, self.validator]):
            return False
            
        # En desarrollo, podemos omitir la validación para facilitar las pruebas
        if ENVIRONMENT == "development":
            logger.warning("Omitiendo validación de firma de Twilio en entorno de desarrollo")
            return True
            
        twilio_signature = request.headers.get('X-Twilio-Signature')
        url = str(request.url)
        form_data = await request.form()
        params = dict(form_data)
        
        return self.validator.validate(url, params, twilio_signature)
    
    async def parse_message(self, request: Request) -> Dict[str, str]:
        try:
            # Obtener datos del formulario
            form_data = await request.form()
            params = dict(form_data)
            
            # Loguear para debugging
            logger.debug(f"Datos recibidos de Twilio: {params}")
            
            # Extraer información del mensaje
            message_body = params.get('Body', '')
            from_number = params.get('From', '').replace('whatsapp:', '')
            to_number = params.get('To', '').replace('whatsapp:', '')
            
            return {
                'body': message_body,
                'from': from_number,
                'to': to_number
            }
        except Exception as e:
            logger.error(f"Error al parsear mensaje de Twilio: {str(e)}")
            logger.error(f"Request headers: {request.headers}")
            logger.error(f"Request method: {request.method}")
            
            # Intentar obtener el cuerpo completo de la solicitud para diagnóstico
            try:
                body = await request.body()
                logger.error(f"Request body: {body}")
            except Exception:
                pass
                
            # Devolver diccionario vacío en caso de error
            return {'body': '', 'from': '', 'to': ''}
    
    async def send_message(self, to: str, body: str) -> Dict[str, Any]:
        try:
            response = self.client.messages.create(
                body=body,
                from_=f"whatsapp:{self.phone_number}",
                to=f"whatsapp:{to}"
            )
            
            return {
                'status': 'success',
                'message_id': response.sid
            }
        except Exception as e:
            logger.error(f"Error al enviar mensaje con Twilio: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Implementación para API oficial de WhatsApp
class OfficialWhatsAppHandler(WhatsAppHandler):
    def __init__(self, api_token: str, phone_number_id: str):
        self.api_token = api_token
        self.phone_number_id = phone_number_id
        self.api_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    async def validate_request(self, request: Request) -> bool:
        # En desarrollo, podemos omitir la validación para facilitar las pruebas
        if ENVIRONMENT == "development":
            logger.warning("Omitiendo validación de firma de WhatsApp API en entorno de desarrollo")
            return True
            
        # La API oficial usa un token de verificación diferente
        # Implementa aquí la validación según la documentación oficial
        return True
    
    async def parse_message(self, request: Request) -> Dict[str, str]:
        try:
            # La estructura de la solicitud de la API oficial es diferente a Twilio
            data = await request.json()
            
            # Loguear para debugging
            logger.debug(f"Datos recibidos de WhatsApp API: {data}")
            
            try:
                # Estructura estándar de webhook de WhatsApp API
                entry = data.get('entry', [])[0]
                changes = entry.get('changes', [])[0]
                value = changes.get('value', {})
                message = value.get('messages', [])[0]
                
                message_body = message.get('text', {}).get('body', '')
                from_number = message.get('from', '')
                
                return {
                    'body': message_body,
                    'from': from_number,
                    'to': self.phone_number_id
                }
            except (IndexError, KeyError) as e:
                logger.error(f"Error al parsear mensaje de WhatsApp API: {str(e)}")
                logger.error(f"Estructura del mensaje: {data}")
                return {'body': '', 'from': '', 'to': ''}
                
        except Exception as e:
            logger.error(f"Error general al parsear mensaje de WhatsApp API: {str(e)}")
            logger.error(f"Request headers: {request.headers}")
            logger.error(f"Request method: {request.method}")
            
            # Intentar obtener el cuerpo completo de la solicitud para diagnóstico
            try:
                body = await request.body()
                logger.error(f"Request body: {body}")
            except Exception:
                pass
                
            # Devolver diccionario vacío en caso de error
            return {'body': '', 'from': '', 'to': ''}
    
    async def send_message(self, to: str, body: str) -> Dict[str, Any]:
        try:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {
                    "body": body
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                return {
                    'status': 'success',
                    'message_id': response.json().get('messages', [{}])[0].get('id', '')
                }
            else:
                logger.error(f"Error al enviar mensaje por WhatsApp API: {response.text}")
                return {
                    'status': 'error',
                    'message': response.text
                }
        except Exception as e:
            logger.error(f"Error general al enviar mensaje con WhatsApp API: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

# Función para obtener el handler adecuado según la configuración
def get_whatsapp_handler() -> WhatsAppHandler:
    if WHATSAPP_PROVIDER == WhatsAppProvider.TWILIO and twilio_client and twilio_validator:
        return TwilioWhatsAppHandler(
            twilio_client,
            twilio_validator,
            TWILIO_PHONE_NUMBER
        )
    elif WHATSAPP_PROVIDER == WhatsAppProvider.OFFICIAL_API and WHATSAPP_API_TOKEN and WHATSAPP_PHONE_NUMBER_ID:
        return OfficialWhatsAppHandler(
            WHATSAPP_API_TOKEN,
            WHATSAPP_PHONE_NUMBER_ID
        )
    else:
        raise HTTPException(
            status_code=503,
            detail=f"Integración con WhatsApp ({WHATSAPP_PROVIDER}) no disponible. Faltan credenciales."
        )

# Inicializar FastAPI
app = FastAPI(
    title="Chatbot UBA - API",
    description="API para el chatbot educativo de la Facultad de Medicina de la UBA",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema RAG (usar variables de entorno)
model_path = os.getenv('MODEL_PATH', 'models/finetuned_model')
embeddings_dir = os.getenv('EMBEDDINGS_DIR', 'data/embeddings')
rag_system = RAGSystem(model_path, embeddings_dir)

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, whatsapp_handler: WhatsAppHandler = Depends(get_whatsapp_handler)):
    """
    Webhook para recibir mensajes de WhatsApp (compatible con Twilio y API oficial).
    
    Args:
        request (Request): Request de FastAPI
        
    Returns:
        Dict: Respuesta con el estado del procesamiento
    """
    try:
        # Para debugging, loguear el método y las cabeceras
        logger.info(f"Webhook recibido - Método: {request.method}")
        logger.info(f"Headers: {request.headers}")
        
        # Para Twilio, devolvemos un TwiML válido aunque ocurra un error
        is_twilio = WHATSAPP_PROVIDER == WhatsAppProvider.TWILIO
        
        try:
            # Validar solicitud
            is_valid = await whatsapp_handler.validate_request(request)
            if not is_valid:
                logger.warning("Solicitud inválida recibida en el webhook")
                raise HTTPException(status_code=403, detail="Solicitud inválida")
                
            # Parsear mensaje
            message = await whatsapp_handler.parse_message(request)
            logger.info(f"Mensaje recibido: {message}")
            
            if not message.get('body'):
                logger.warning("Mensaje recibido sin cuerpo")
                if is_twilio:
                    return {"message": "No message body found"}
                return {"status": "ignored", "message": "No se encontró contenido en el mensaje"}
                
            # Generar respuesta usando RAG
            try:
                result = rag_system.process_query(message['body'])
                response_text = result['response']
            except Exception as e:
                logger.error(f"Error al procesar consulta RAG: {str(e)}")
                response_text = "Lo siento, tuve un problema procesando tu mensaje. Por favor, intenta de nuevo."
            
            # Enviar respuesta por WhatsApp
            send_result = await whatsapp_handler.send_message(
                message['from'],
                response_text
            )
            
            # Si es Twilio, devolvemos una respuesta compatible
            if is_twilio:
                return {"message": "Message processed successfully"}
                
            # Si no es Twilio, devolvemos detalles completos
            return {
                "status": "success",
                "message": "Respuesta enviada",
                "details": send_result
            }
            
        except Exception as e:
            logger.error(f"Error en webhook de WhatsApp: {str(e)}")
            
            # Si es Twilio, devolvemos una respuesta compatible
            if is_twilio:
                return {"message": "Error processing message"}
                
            # Si no es Twilio, devolvemos error estándar
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error general en webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-message")
async def test_message():
    """
    Envía un mensaje de prueba al número de WhatsApp configurado en MY_PHONE_NUMBER.
    
    IMPORTANTE PARA TWILIO SANDBOX:
    - En modo sandbox, Twilio SOLO permite enviar mensajes a números que se hayan unido previamente.
    - Para unirse al sandbox, el usuario debe enviar un mensaje específico (ejemplo: "join apple-brown") 
      desde su WhatsApp al número de Twilio.
    - El código exacto de unión se encuentra en la consola de Twilio en la sección de WhatsApp Sandbox.
    - Si no recibes el mensaje de prueba, verifica que hayas enviado el mensaje de unión correctamente.
    
    Returns:
        dict: Resultado del envío del mensaje de prueba
    """
    if not MY_PHONE_NUMBER:
        return {"error": "No se ha configurado MY_PHONE_NUMBER en las variables de entorno"}
    
    try:
        if ENVIRONMENT == "development":
            # En desarrollo (Twilio)
            logging.info(f"Enviando mensaje de prueba vía Twilio a {MY_PHONE_NUMBER}")
            
            if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
                return {"error": "Faltan credenciales de Twilio en las variables de entorno"}
            
            # Advertencia sobre sandbox de Twilio
            logging.warning(
                "MODO SANDBOX DE TWILIO ACTIVADO: Asegúrate de que el número "
                f"{MY_PHONE_NUMBER} haya enviado el mensaje de unión al sandbox "
                "desde la consola de Twilio. De lo contrario, NO recibirás mensajes."
            )
            
            # Crear cliente de Twilio
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            
            # Enviar mensaje
            message = client.messages.create(
                from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
                body="¡Hola! Este es un mensaje de prueba del Chatbot UBA Medicina. Si lo recibes, la configuración es correcta.",
                to=f"whatsapp:{MY_PHONE_NUMBER}"
            )
            
            return {
                "success": True,
                "message_sid": message.sid,
                "status": message.status,
                "note": "Si no recibes el mensaje, verifica que hayas unido tu número al sandbox de Twilio"
            }
        
        elif ENVIRONMENT == "production":
            # En producción (API oficial de WhatsApp)
            logging.info(f"Enviando mensaje de prueba vía API oficial de WhatsApp a {MY_PHONE_NUMBER}")
            
            if not WHATSAPP_API_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
                return {"error": "Faltan credenciales de WhatsApp Business API en las variables de entorno"}
            
            # Configurar headers para la API de WhatsApp
            headers = {
                "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
                "Content-Type": "application/json"
            }
            
            # Preparar datos del mensaje
            data = {
                "messaging_product": "whatsapp",
                "to": MY_PHONE_NUMBER,
                "type": "text",
                "text": {
                    "body": "¡Hola! Este es un mensaje de prueba del Chatbot UBA Medicina. Si lo recibes, la configuración es correcta."
                }
            }
            
            # Enviar solicitud a la API de WhatsApp
            response = requests.post(
                f"https://graph.facebook.com/v13.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
                headers=headers,
                json=data
            )
            
            # Verificar respuesta
            if response.status_code == 200:
                return {
                    "success": True,
                    "message_id": response.json().get("messages", [{}])[0].get("id"),
                    "status": "sent"
                }
            else:
                return {
                    "success": False,
                    "error": f"Error al enviar mensaje: {response.status_code}",
                    "details": response.json()
                }
    
    except Exception as e:
        logging.error(f"Error al enviar mensaje de prueba: {str(e)}")
        return {"error": f"Error al enviar mensaje de prueba: {str(e)}"}

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del servicio."""
    whatsapp_available = (
        (WHATSAPP_PROVIDER == WhatsAppProvider.TWILIO and twilio_client is not None) or
        (WHATSAPP_PROVIDER == WhatsAppProvider.OFFICIAL_API and 
         WHATSAPP_API_TOKEN is not None and WHATSAPP_PHONE_NUMBER_ID is not None)
    )
    
    status_info = {
        "status": "healthy", 
        "environment": ENVIRONMENT,
        "whatsapp_provider": WHATSAPP_PROVIDER,
        "whatsapp_available": whatsapp_available,
        "test_number_configured": MY_PHONE_NUMBER is not None,
        "model_path": model_path,
        "embeddings_dir": embeddings_dir
    }
    
    # Añadir información sobre requisitos del sandbox de Twilio
    if ENVIRONMENT == "development" and WHATSAPP_PROVIDER == WhatsAppProvider.TWILIO:
        status_info["twilio_sandbox_info"] = {
            "warning": "Estás usando el sandbox de Twilio para WhatsApp",
            "join_requirement": "El número receptor debe enviar un mensaje de unión al sandbox",
            "instructions": "Busca la sección 'WhatsApp Sandbox' en tu panel de Twilio para ver el código exacto",
            "example": "El usuario debe enviar algo como 'join apple-brown' al número de Twilio"
        }
    
    return status_info

@app.post("/chat")
async def chat_endpoint(message: Dict[str, str]):
    """
    Endpoint para probar el chatbot sin WhatsApp.
    
    Args:
        message (Dict[str, str]): Diccionario con el mensaje
        
    Returns:
        Dict: Respuesta del chatbot
    """
    try:
        query = message.get("message")
        if not query:
            raise HTTPException(status_code=400, detail="Mensaje no proporcionado")
            
        result = rag_system.process_query(query)
        return result
        
    except Exception as e:
        logger.error(f"Error en endpoint de chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Función principal para ejecutar el servidor."""
    # Configuración del servidor desde variables de entorno
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    
    logger.info(f"Iniciando servidor en {host}:{port}")
    logger.info(f"Entorno: {ENVIRONMENT}")
    logger.info(f"Proveedor de WhatsApp: {WHATSAPP_PROVIDER}")
    logger.info(f"Ruta del modelo: {model_path}")
    logger.info(f"Ruta de embeddings: {embeddings_dir}")
    if MY_PHONE_NUMBER:
        logger.info(f"Número de prueba configurado: {MY_PHONE_NUMBER}")
    
    # Iniciar servidor
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 