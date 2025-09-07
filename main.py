import os
import logging
import json
import requests
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response, Header
from dotenv import load_dotenv
from core.app_manager import app_manager
from services.session_service import session_service
from services.metrics_service import metrics_service
import uvicorn
import re

# Cargar variables de entorno
load_dotenv()

# Configuración de logging - usar INFO en producción, DEBUG en desarrollo
log_level = logging.DEBUG if app_manager.environment == 'development' else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Iniciando en entorno: {app_manager.environment}")

# Inicializar FastAPI
app = FastAPI(
    title="Chatbot UBA - API",
    description="API para el chatbot educativo de la Facultad de Medicina de la UBA",
    version="1.0.0"
)


# Middleware para logging y debugging de webhooks
# Middleware para logging y debugging
@app.middleware("http")
async def telegram_webhook_middleware(request: Request, call_next):
    """Middleware para loggear todas las requests."""
    
    # Logging de la request entrante
    logger.info(f"Request recibida: {request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    
    # Verificación especial para webhook de Telegram
    if request.url.path == "/webhook/telegram":
        try:
            # Para requests GET (verificación de webhook)
            if request.method == "GET":
                logger.info("Verificación GET del webhook de Telegram")
            
            # Para requests POST (mensajes)
            elif request.method == "POST":
                # Log básico sin consumir el body
                secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
                logger.info(f"Webhook POST de Telegram - Secret token presente: {bool(secret_token)}")
                
        except Exception as e:
            logger.error(f"Error en middleware de Telegram: {str(e)}")
    
    # Procesar la request
    response = await call_next(request)
    
    logger.info(f"Response: {response.status_code}")
    return response

@app.on_event("startup")
async def startup_event():
    """Inicializa componentes del sistema al iniciar el servidor."""
    try:
        logger.info("Inicializando componentes del sistema...")
        
        # Pre-inicializar sistema RAG para verificar configuración
        app_manager.get_rag_system()
        logger.info("✅ Sistema RAG pre-inicializado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error al inicializar componentes: {str(e)}")
        # No fallar el startup, permitir que el servidor inicie y maneje errores dinámicamente


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza en apagado del servicio."""
    try:
        # Detener sweeper de sesiones para apagado limpio
        session_service.stop()
        logger.info("SessionService detenido correctamente")
    except Exception as e:
        logger.warning(f"Error al detener SessionService: {str(e)}")

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request):
    """
    Webhook para recibir mensajes de Telegram.
    
    Args:
        request (Request): Request de FastAPI
        
    Returns:
        Dict: Respuesta con el estado del procesamiento
    """
    try:
        logger.info(f"Webhook de Telegram recibido - Método: {request.method}")
        logger.info(f"Headers: {request.headers}")
        
        # Obtener el handler de Telegram
        telegram_handler = get_telegram_handler()
        
        # Validar webhook
        if not await telegram_handler.validate_webhook(request):
            raise HTTPException(status_code=403, detail="Webhook no autorizado")
        
        # Parsear mensaje
        message_data = await telegram_handler.parse_message(request)
        
        if not message_data.get("body"):
            return {"status": "ignored", "message": "Mensaje vacío o no procesable"}
        
        # Extraer datos del mensaje
        message_body = message_data["body"]
        user_id = message_data["from"]
        chat_id = message_data["chat_id"]
        profile_name = message_data["profile_name"]
        
        logger.info(f"Mensaje recibido de {profile_name} ({user_id}): {message_body}")
        
        # Verificar que RAG esté disponible
        if not app_manager.is_rag_ready():
            logger.error("Sistema RAG no disponible")
            return {
                "status": "error", 
                "message": "Sistema RAG no disponible"
            }
        
        # Enviar acción de typing
        await telegram_handler.send_typing_action(chat_id)
        
        try:
            # Procesar con RAG
            result = app_manager.get_rag_system().process_query(
                message_body, 
                user_id=user_id,
                user_name=profile_name
            )
            response_text = result["response"]
            
            # Enviar respuesta a Telegram
            send_result = await telegram_handler.send_message(
                chat_id,
                response_text
            )
            
            return {
                "status": "success",
                "from": user_id,
                "chat_id": chat_id,
                "message": message_body,
                "response": response_text
            }
            
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {str(e)}")
            return {"status": "error", "message": str(e)}
        
    except Exception as e:
        logger.error(f"Error en webhook de Telegram: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webhook/telegram")
async def verify_telegram_webhook(request: Request):
    """Endpoint para verificar el webhook de Telegram."""
    try:
        telegram_handler = get_telegram_handler()
        webhook_info = await telegram_handler.get_webhook_info()
        
        return {
            "status": "success",
            "webhook_info": webhook_info
        }
        
    except Exception as e:
        logger.error(f"Error al verificar webhook de Telegram: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-telegram")
async def test_telegram():
    """Endpoint para probar la conexión con Telegram."""
    logger.info("Endpoint de prueba /test-telegram llamado")
    
    try:
        if not app_manager.is_telegram_ready():
            return {"status": "error", "message": "TELEGRAM_BOT_TOKEN no configurado"}
    
        telegram_handler = app_manager.get_telegram_handler()
        
        # Test básico: obtener información del bot
        bot_info = await telegram_handler.get_me()
        
        if bot_info.get("ok"):
            bot_data = bot_info.get("result", {})
            response_data = {
                "status": "success",
                "message": "Conexión con Telegram exitosa",
                "bot_info": {
                    "id": bot_data.get("id"),
                    "username": bot_data.get("username"),
                    "first_name": bot_data.get("first_name"),
                    "can_join_groups": bot_data.get("can_join_groups"),
                    "can_read_all_group_messages": bot_data.get("can_read_all_group_messages"),
                    "supports_inline_queries": bot_data.get("supports_inline_queries")
                }
            }
            
            # Si hay un admin user ID configurado, enviar mensaje de prueba
            if TELEGRAM_ADMIN_USER_ID:
                test_result = await telegram_handler.send_message(
                    TELEGRAM_ADMIN_USER_ID,
                    "🤖 Este es un mensaje de prueba desde el endpoint /test-telegram.\n\n✅ El bot está funcionando correctamente."
                )
                
                if test_result.get('status') == 'success':
                    response_data["test_message"] = "Mensaje de prueba enviado al admin"
                else:
                    response_data["test_message_error"] = test_result.get('message')
            
            return response_data
        else:
            return {
                "status": "error", 
                "message": "Error al conectar con Telegram",
                "details": bot_info
            }
    except Exception as e:
        logger.error(f"Error en test-telegram: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.post("/telegram/setup-webhook")
async def setup_telegram_webhook(webhook_url: str):
    """Configura el webhook de Telegram."""
    try:
        telegram_handler = get_telegram_handler()
        result = await telegram_handler.set_webhook(webhook_url)
        
        return {
            "status": "success" if result.get("ok") else "error",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error al configurar webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/telegram/delete-webhook")
async def delete_telegram_webhook():
    """Elimina el webhook de Telegram."""
    try:
        telegram_handler = get_telegram_handler()
        result = await telegram_handler.delete_webhook()
        
        return {
            "status": "success" if result.get("ok") else "error",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error al eliminar webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado del servicio."""
    try:
        # Obtener estado del sistema desde AppManager
        system_status = app_manager.get_system_status()
        rag_status = "initialized" if system_status["rag_ready"] else "not_initialized"
        telegram_available = system_status["telegram_configured"]
        
        status_info = {
            "status": "healthy",
            "environment": system_status["environment"],
            "rag_status": rag_status,
            "telegram_available": telegram_available,
            "database": "PostgreSQL con pgvector"
        }
        
        if telegram_available:
            status_info["telegram_config"] = {
                "bot_token": "configured" if app_manager.telegram_bot_token else "missing",
                "webhook_secret": "configured" if app_manager.telegram_webhook_secret else "missing",
                "admin_user_id": "configured" if app_manager.telegram_admin_user_id else "missing"
            }
        
        # Añadir estadísticas de sesiones
        status_info["session_stats"] = session_service.get_session_stats()
        status_info["router_metrics"] = metrics_service.get_stats()
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

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
            # Verificar disponibilidad del sistema RAG
            if not app_manager.is_rag_ready():
                logger.error("Sistema RAG no disponible")
                return {
                    "query": query,
                    "response": "Lo siento, el sistema está en mantenimiento. Por favor, intenta más tarde.",
                    "relevant_chunks": [],
                    "sources": []
                }
            
            # Obtener respuesta del sistema RAG
            response = app_manager.get_rag_system().process_query(query)
            
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

# Legacy endpoint para compatibilidad
@app.post("/api/message")
async def receive_legacy_message_redirect(request: Request):
    """Endpoint legacy que redirige a Telegram."""
    logger.warning("Endpoint /api/message es legacy. Use /webhook/telegram")
    return {
        "status": "deprecated",
        "message": "Este endpoint está deprecado. Use /webhook/telegram para Telegram Bot API"
    }

@app.get("/test-webhook")
async def test_webhook():
    """Endpoint legacy que redirige al test de Telegram."""
    logger.warning("Endpoint de prueba /test-webhook llamado - redirigiendo a /test-telegram")
    
    # Redirigir al endpoint actual de Telegram
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/test-telegram", status_code=302)

@app.get("/metrics")
async def get_metrics(x_api_key: str = Header(default=None)):
    """Métricas JSON protegidas por X-API-Key (env METRICS_API_KEY)."""
    api_key = os.getenv('METRICS_API_KEY')
    if not api_key:
        raise HTTPException(status_code=503, detail="Métricas no habilitadas")
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(status_code=403, detail="No autorizado")
    return metrics_service.get_stats()

def main():
    """Función principal para ejecutar el servidor."""
    try:
        # Obtener configuración del servidor
        host = os.getenv('HOST', '0.0.0.0')
        
        # En Cloud Run, la variable PORT es establecida por la plataforma
        # https://cloud.google.com/run/docs/container-contract#port
        port = int(os.getenv('PORT', 8080))
        
        logger.info(f"Iniciando servidor en {host}:{port}")
        logger.info(f"Entorno: {app_manager.environment}")
        logger.info("Sistema configurado para usar PostgreSQL con pgvector")
        logger.info(f"GCS Bucket: {os.getenv('GCS_BUCKET_NAME', 'No configurado')}")
        logger.info(f"Telegram disponible: {app_manager.is_telegram_ready()}")
        
        # Iniciar servidor
        uvicorn.run("main:app", host=host, port=port, reload=(app_manager.environment == 'development'))
    except Exception as e:
        logger.error(f"Error al iniciar servidor: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 