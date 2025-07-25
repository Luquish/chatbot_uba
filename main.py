import os
import logging
import json
import requests
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from dotenv import load_dotenv
from rag_system import RAGSystem
from handlers.whatsapp_handler import WhatsAppHandler
import uvicorn
import re

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,  # Cambiar a DEBUG para ver todos los mensajes
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determinar el entorno (desarrollo o producción)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
logger.info(f"Iniciando en entorno: {ENVIRONMENT}")

# Configuración de WhatsApp Business API
WHATSAPP_API_TOKEN = os.getenv('WHATSAPP_API_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID')
WHATSAPP_WEBHOOK_VERIFY_TOKEN = os.getenv('WHATSAPP_WEBHOOK_VERIFY_TOKEN')

# Número de teléfono para pruebas
MY_PHONE_NUMBER = os.getenv('MY_PHONE_NUMBER')

# Inicialización de variables globales
rag_system = None
rag_initialized = False

# Función para obtener el handler de WhatsApp
def get_whatsapp_handler() -> WhatsAppHandler:
    if all([
        WHATSAPP_API_TOKEN,
        WHATSAPP_PHONE_NUMBER_ID,
        WHATSAPP_BUSINESS_ACCOUNT_ID
    ]):
        return WhatsAppHandler(
            WHATSAPP_API_TOKEN,
            WHATSAPP_PHONE_NUMBER_ID,
            WHATSAPP_BUSINESS_ACCOUNT_ID
        )
    else:
        raise HTTPException(
            status_code=503,
            detail="Integración con WhatsApp no disponible. Faltan credenciales."
        )

# Inicializar FastAPI
app = FastAPI(
    title="Chatbot UBA - API",
    description="API para el chatbot educativo de la Facultad de Medicina de la UBA",
    version="1.0.0"
)


# Middleware para manejo directo de WebHooks de WhatsApp en producción
if ENVIRONMENT == "production":
    @app.middleware("http")
    async def whatsapp_webhook_middleware(request: Request, call_next):
        """
        Middleware para manejar la verificación de webhooks de WhatsApp en producción.
        Solo se activa en entorno de producción para permitir la comunicación directa
        entre WhatsApp Business API y Cloud Run, sin necesidad de Glitch como intermediario.
        """
        # Solo interceptar peticiones GET al endpoint del webhook (verificación)
        if request.url.path == "/webhook/whatsapp" and request.method == "GET":
            logger.info("Interceptando solicitud de verificación de webhook directa")
            # Obtener parámetros de verificación
            params = request.query_params
            mode = params.get("hub.mode")
            token = params.get("hub.verify_token")
            challenge = params.get("hub.challenge")
            
            # Obtener token de verificación configurado
            verify_token = WHATSAPP_WEBHOOK_VERIFY_TOKEN
            
            # Verificar modo y token
            if mode == "subscribe" and token == verify_token:
                logger.info(f"Webhook verificado con éxito en middleware: {challenge}")
                # Responder con el challenge directamente
                return Response(content=challenge, media_type="text/plain")
            else:
                logger.warning(f"Verificación fallida del webhook. Token esperado: {verify_token}, recibido: {token}")
                return Response(status_code=403, content="Forbidden")
                
        # Para todas las demás solicitudes, continuar con el flujo normal
        return await call_next(request)
        
    logger.info("Middleware para manejo directo de webhooks de WhatsApp activado (modo producción)")
else:
    logger.info("Ejecutando en modo desarrollo - Webhooks redirigidos desde Glitch")

@app.on_event("startup")
async def startup_event():
    """Inicializa el sistema RAG al iniciar el servidor."""
    global rag_system, rag_initialized
    try:
        logger.info("Iniciando sistema RAG...")
        
        # Detectar si estamos en Docker o local
        # Si /app existe como directorio, estamos en Docker
        if os.path.exists('/app'):
            embeddings_dir = '/app/data/embeddings'
        else:
            # Estamos ejecutando localmente
            embeddings_dir = 'data/embeddings'
            
        logger.info(f"Utilizando directorio de embeddings: {embeddings_dir}")
        rag_system = RAGSystem(
            embeddings_dir=embeddings_dir
        )
        rag_initialized = True
        logger.info("Sistema RAG inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar sistema RAG: {str(e)}")
        raise e

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Webhook para recibir mensajes de WhatsApp.
    En producción: Recibe directamente mensajes de la API de WhatsApp Business.
    En desarrollo: Recibe mensajes redireccionados desde Glitch.
    
    Args:
        request (Request): Request de FastAPI
        
    Returns:
        Dict: Respuesta con el estado del procesamiento
    """
    try:
        # Para debugging, loguear el método y las cabeceras
        logger.info(f"Webhook recibido - Método: {request.method}")
        logger.info(f"Headers: {request.headers}")
        
        # Obtener el cuerpo de la solicitud
        body = await request.body()
        
        # Validar la firma del webhook en producción
        if ENVIRONMENT == "production":
            signature = request.headers.get("x-hub-signature-256", "")
            if not signature:
                logger.error("Firma no encontrada en headers")
                raise HTTPException(status_code=403, detail="Firma no encontrada")

            # Calcular y verificar firma
            import hmac
            import hashlib

            expected_signature = hmac.new(
                WHATSAPP_WEBHOOK_VERIFY_TOKEN.encode(),
                body,
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(f"sha256={expected_signature}", signature):
                logger.error("Firma inválida")
                raise HTTPException(status_code=403, detail="Firma inválida")
        
        # Parsear el cuerpo JSON
        try:
            data = await request.json()
            logger.debug(f"Datos recibidos: {json.dumps(data, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error al parsear JSON: {str(e)}")
            raise HTTPException(status_code=400, detail="JSON inválido")
        
        # Verificar si es un mensaje
        if "object" not in data:
            logger.warning("Objeto no encontrado en la solicitud")
            return {"status": "ignored"}
            
        if data["object"] != "whatsapp_business_account":
            logger.warning(f"Objeto no esperado: {data['object']}")
            return {"status": "ignored"}
            
        # Procesar entrada
        try:
            entry = data["entry"][0]
            changes = entry["changes"][0]
            value = changes["value"]
            
            # Si tenemos un mensaje o cualquier evento, procesamos
            if "messages" in value:
                message = value.get("messages", [])[0]
                message_body = message.get("text", {}).get("body", "")
                from_number = message.get("from", "")
                message_id = message.get("id", "")
                
                # Obtener el nombre del perfil si está disponible
                contacts = value.get("contacts", [])
                profile_name = contacts[0].get("profile", {}).get("name", "") if contacts else ""
                logger.info(f"Nombre del perfil: {profile_name}")
                
                # Extraer ID del número de teléfono de negocio desde metadata
                business_phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
                
                logger.info(f"Webhook completo - De: {from_number}, Nombre: {profile_name}, Mensaje: '{message_body}'")
                
                # Procesar el mensaje usando RAG
                if not rag_initialized:
                    logger.error("Sistema RAG no inicializado")
                    return {
                        "status": "error", 
                        "message": "Sistema RAG no inicializado"
                    }
                
                try:
                    # Normalizar el número del remitente para WhatsApp
                    whatsapp_handler = get_whatsapp_handler()
                    normalized_from = whatsapp_handler.normalize_phone_number(from_number)
                    
                    # Procesar con RAG
                    result = rag_system.process_query(
                        message_body, 
                        user_id=normalized_from,
                        user_name=profile_name
                    )
                    response_text = result["response"]
                    
                    # Enviar respuesta a WhatsApp
                    send_result = await whatsapp_handler.send_message(
                        normalized_from,
                        response_text
                    )
                    
                    return {
                        "status": "success",
                        "from": normalized_from,
                        "message": message_body,
                        "response": response_text
                    }
                except Exception as e:
                    logger.error(f"Error al procesar mensaje: {str(e)}")
                    return {"status": "error", "message": str(e)}
                
            elif "statuses" in value:
                status = value.get("statuses", [])[0]
                status_type = status.get("status")
                message_id = status.get("id")
                recipient = status.get("recipient_id")
                
                if status_type == "sent":
                    logger.debug(f"✓ Mensaje {message_id} enviado a {recipient}")
                elif status_type == "delivered":
                    logger.debug(f"✓✓ Mensaje {message_id} entregado a {recipient}")
                elif status_type == "read":
                    logger.debug(f"✓✓✓ Mensaje {message_id} leído por {recipient}")
                else:
                    logger.debug(f"Estado desconocido para mensaje {message_id}: {status_type}")
                
                return {"status": "success", "message": f"Estado {status_type} procesado"}
            else:
                logger.warning("Webhook recibido sin mensajes ni estados")
                return {"status": "ignored", "message": "Webhook sin contenido procesable"}
                
        except (KeyError, IndexError) as e:
            logger.error(f"Error al procesar entrada: {str(e)}")
            return {"status": "error", "message": "Formato de entrada inválido"}
        
        return {"status": "ignored", "message": "Evento no procesable"}
            
    except Exception as e:
        logger.error(f"Error en webhook de WhatsApp: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    """
    Endpoint para verificar el webhook de WhatsApp.
    """
    try:
        # Obtener parámetros de la consulta
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        
        # Loguear para debugging
        logger.info(f"Verificación de webhook recibida:")
        logger.info(f"Mode: {mode}")
        logger.info(f"Token recibido: {token}")
        logger.info(f"Token esperado: {WHATSAPP_WEBHOOK_VERIFY_TOKEN}")
        logger.info(f"Challenge: {challenge}")
        
        # Verificar modo y token
        if mode and token and mode == "subscribe" and token == WHATSAPP_WEBHOOK_VERIFY_TOKEN:
            if not challenge:
                logger.error("Challenge no encontrado")
                return {"status": "error", "message": "No challenge found"}
                
            logger.info(f"Verificación exitosa, devolviendo challenge: {challenge}")
            return int(challenge)
            
        logger.error("Verificación fallida - Token o modo inválidos")
        raise HTTPException(status_code=403, detail="Forbidden")
            
    except Exception as e:
        logger.error(f"Error en verificación de webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-message")
async def test_message():
    """
    Envía un mensaje de prueba al número de WhatsApp configurado en MY_PHONE_NUMBER.
    
    Returns:
        dict: Resultado del envío del mensaje de prueba
    """
    if not MY_PHONE_NUMBER:
        logger.error("No se ha configurado MY_PHONE_NUMBER en las variables de entorno")
        return {"error": "No se ha configurado MY_PHONE_NUMBER en las variables de entorno"}
    
    try:
        # Verificar credenciales
        if not WHATSAPP_API_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            return {"error": "Faltan credenciales de WhatsApp Business API en las variables de entorno"}
        
        # Formatear el número de teléfono según el formato de la API de WhatsApp
        # (sin el signo '+', solo dígitos)
        formatted_phone = MY_PHONE_NUMBER
        # Eliminar el '+' si existe y cualquier otro caracter no numérico
        formatted_phone = re.sub(r'[^0-9]', '', formatted_phone)
            
        logger.info(f"Enviando mensaje de prueba a: {MY_PHONE_NUMBER} (formateado: {formatted_phone})")
        
        # Configurar headers para la API de WhatsApp
        headers = {
            "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Preparar datos del mensaje
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": formatted_phone,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": "¡Hola! Este es un mensaje de prueba del Chatbot UBA Medicina. Si lo recibes, la configuración es correcta."
            }
        }
        
        # Enviar solicitud a la API de WhatsApp
        response = requests.post(
            f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=data
        )
        
        # Verificar respuesta
        response_data = response.json()
        
        if response.status_code == 200:
            return {
                "success": True,
                "message_id": response_data.get("messages", [{}])[0].get("id"),
                "status": "sent",
                "to": MY_PHONE_NUMBER,
                "whatsapp_api_response": response_data
            }
        else:
            error_message = response_data.get('error', {}).get('message', 'Error desconocido')
            error_code = response_data.get('error', {}).get('code', '')
            return {
                "success": False,
                "error": f"Error al enviar mensaje: {error_message} (Código: {error_code})",
                "to": MY_PHONE_NUMBER,
                "details": response_data
            }
    
    except Exception as e:
        logging.error(f"Error al enviar mensaje de prueba: {str(e)}")
        return {"error": f"Error al enviar mensaje de prueba: {str(e)}"}

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del servicio."""
    whatsapp_available = (
        WHATSAPP_API_TOKEN is not None and 
        WHATSAPP_PHONE_NUMBER_ID is not None and
        WHATSAPP_BUSINESS_ACCOUNT_ID is not None and
        WHATSAPP_WEBHOOK_VERIFY_TOKEN is not None
    )
    
    status_info = {
        "status": "healthy", 
        "environment": ENVIRONMENT,
        "whatsapp_available": whatsapp_available,
        "test_number_configured": MY_PHONE_NUMBER is not None,
        "model_path": "data/embeddings",
        "embeddings_dir": "data/embeddings"
    }
    
    # Añadir información sobre la configuración de WhatsApp
    if whatsapp_available:
        status_info["whatsapp_config"] = {
            "phone_number_id": WHATSAPP_PHONE_NUMBER_ID,
            "business_account_id": WHATSAPP_BUSINESS_ACCOUNT_ID,
            "webhook_verify_token": "configured" if WHATSAPP_WEBHOOK_VERIFY_TOKEN else "missing"
        }
    
    return status_info

@app.post("/chat")
async def chat_endpoint(message: Dict[str, str]):
    """
    Endpoint para el chat web que usa el sistema RAG mejorado.
    """
    try:
        query = message.get('message', '').strip()
        if not query:
            raise HTTPException(status_code=400, detail="Mensaje vacío")
            
        logger.info(f"Consulta recibida: {query}")
        
        # Procesar la consulta con el sistema RAG
        try:
            # Usar la instancia global de RAG si existe
            global rag_system
            if not rag_initialized:
                logger.error("Sistema RAG no inicializado")
                return {
                    "query": query,
                    "response": "Lo siento, el sistema está en mantenimiento. Por favor, intenta más tarde.",
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Obtener respuesta del sistema RAG
            response = rag_system.process_query(query)
            
            # Verificar si se encontraron chunks relevantes
            if not response.get('relevant_chunks'):
                logger.warning("No se encontraron chunks relevantes")
                return {
                    "query": query,
                    "response": "Lo siento, no encontré información relevante para responder tu pregunta. ¿Podrías reformularla?",
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Extraer fuentes únicas
            sources = list(set(chunk['filename'].replace('.pdf', '') 
                             for chunk in response['relevant_chunks']))
            
            # Logging de resultados
            logger.info(f"Chunks relevantes encontrados: {len(response['relevant_chunks'])}")
            logger.info(f"Fuentes utilizadas: {sources}")
            
            return {
                "query": query,
                "response": response['response'],
                "relevant_chunks": response['relevant_chunks'],
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error al procesar consulta con RAG: {str(e)}", exc_info=True)
            return {
                "query": query,
                "response": "Lo siento, ocurrió un error al procesar tu consulta. Por favor, intenta de nuevo.",
                "relevant_chunks": [],
                "sources": []
            }
            
    except Exception as e:
        logger.error(f"Error general en endpoint /chat: {str(e)}", exc_info=True)
        return {
            "query": query if 'query' in locals() else "",
            "response": "Error interno del servidor. Por favor, intenta más tarde.",
            "relevant_chunks": [],
            "sources": []
        }

@app.post("/api/whatsapp/message")
async def receive_whatsapp_message(request: Request):
    """
    Endpoint para recibir mensajes de WhatsApp redireccionados desde Glitch.
    Se mantiene para compatibilidad en entorno de desarrollo.
    """
    # Agregar log de entrada para confirmar que se está llamando al endpoint
    logger.info("======= MENSAJE RECIBIDO EN /api/whatsapp/message =======")
    
    if ENVIRONMENT == "production":
        logger.warning("Endpoint /api/whatsapp/message llamado en producción. Considere actualizar la configuración para usar /webhook/whatsapp directamente.")
    
    # En vez de duplicar lógica, redirigimos internamente al endpoint principal
    return await whatsapp_webhook(request)

@app.get("/test-webhook")
async def test_webhook():
    """Endpoint para probar la conexión directamente."""
    logger.warning("Endpoint de prueba /test-webhook llamado")
    
    try:
        # Verificar que podemos enviar mensajes
        if MY_PHONE_NUMBER:
            try:
                formatted_phone = re.sub(r'[^0-9]', '', MY_PHONE_NUMBER)
                
                whatsapp_handler = get_whatsapp_handler()
                result = await whatsapp_handler.send_message(
                    formatted_phone,
                    "Este es un mensaje de prueba directo desde el endpoint /test-webhook."
                )
                
                if result.get('status') == 'success':
                    return {
                        "status": "success",
                        "message": "Mensaje de prueba enviado directamente",
                        "to": formatted_phone,
                        "details": result
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Error al enviar mensaje de prueba",
                        "details": result
                    }
            except Exception as e:
                logger.error(f"Error al enviar mensaje directo: {str(e)}", exc_info=True)
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": "No hay número de teléfono configurado para pruebas"}
    except Exception as e:
        logger.error(f"Error en test-webhook: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

def main():
    """Función principal para ejecutar el servidor."""
    try:
        # Obtener configuración del servidor
        host = os.getenv('HOST', '0.0.0.0')
        
        # En Cloud Run, la variable PORT es establecida por la plataforma
        # https://cloud.google.com/run/docs/container-contract#port
        port = int(os.getenv('PORT', 8080))
        
        logger.info(f"Iniciando servidor en {host}:{port}")
        logger.info(f"Entorno: {ENVIRONMENT}")
        logger.info(f"Ruta de embeddings: {os.getenv('EMBEDDINGS_DIR', 'data/embeddings')}")
        logger.info(f"GCS Bucket: {os.getenv('GCS_BUCKET_NAME', 'No configurado')}")
        logger.info(f"Número de prueba configurado: {MY_PHONE_NUMBER}")
        
        # Iniciar servidor
        uvicorn.run("main:app", host=host, port=port, reload=(ENVIRONMENT == 'development'))
    except Exception as e:
        logger.error(f"Error al iniciar servidor: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 