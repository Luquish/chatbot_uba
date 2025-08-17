import os
import logging
import json
import requests
import hmac
import hashlib
import asyncio
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logger = logging.getLogger(__name__)


class TelegramHandler:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Debug logging
        logger.info(f"TelegramHandler inicializado con token: {bot_token[:10]}...")
        logger.info(f"API URL: {self.api_url}")
        
    async def validate_webhook(self, request: Request) -> bool:
        """Valida la autenticidad del webhook de Telegram."""
        if ENVIRONMENT == "development":
            logger.warning("Omitiendo validación de webhook en desarrollo")
            return True
            
        # Obtener la firma del webhook
        signature = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not signature:
            logger.error("No se encontró firma en el header")
            return False
            
        # Verificar la firma (logging para debug)
        logger.info(f"Signature recibida: {signature}")
        logger.info(f"Webhook secret configurado: {self.webhook_secret}")
        
        if signature != self.webhook_secret:
            logger.error(f"Firma de webhook inválida. Esperado: {self.webhook_secret}, Recibido: {signature}")
            return False
            
        logger.info("Webhook validado correctamente")
        return True
    
    async def parse_message(self, request: Request) -> Dict[str, str]:
        """Parsea el mensaje recibido de Telegram."""
        try:
            data = await request.json()
            logger.debug(f"Datos recibidos de Telegram API: {data}")
            
            # Verificar si es un mensaje
            if "message" not in data:
                logger.warning("Update sin mensaje")
                return {"body": "", "from": "", "to": ""}
                
            message = data["message"]
            user = message.get("from", {})
            
            # Extraer información del mensaje
            message_text = message.get("text", "")
            user_id = str(user.get("id", ""))
            first_name = user.get("first_name", "")
            last_name = user.get("last_name", "")
            username = user.get("username", "")
            chat_id = str(message.get("chat", {}).get("id", ""))
            chat_type = message.get("chat", {}).get("type", "")
            
            profile_name = f"{first_name} {last_name}".strip() or username or f"Usuario {user_id}"
            
            return {
                "body": message_text,
                "from": user_id,
                "to": str(self.bot_token.split(':')[0]),  # Bot ID
                "message_id": str(message.get("message_id", "")),
                "timestamp": str(message.get("date", "")),
                "profile_name": profile_name,
                "username": username,
                "chat_id": chat_id,
                "chat_type": chat_type
            }
            
        except Exception as e:
            logger.error(f"Error al parsear mensaje de Telegram: {str(e)}")
            return {"body": "", "from": "", "to": ""}
    
    async def send_message(self, chat_id: str, text: str) -> Dict[str, Any]:
        """Envía un mensaje a través de Telegram."""
        try:
            logger.info(f"Enviando mensaje a {chat_id}: '{text[:100]}...'")
            
            # Usar la API de Telegram directamente
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",  # Permite formato básico HTML
                "disable_web_page_preview": True
            }
            
            url = f"{self.api_url}/sendMessage"
            logger.info(f"URL de envío: {url}")
            logger.info(f"Payload: {payload}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=30
            )
            
            logger.info(f"Status code de respuesta: {response.status_code}")
            logger.info(f"Response text: {response.text}")
            
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get("ok"):
                message_id = response_data["result"]["message_id"]
                logger.info(f"Mensaje enviado exitosamente. ID: {message_id}")
                return {
                    "status": "success",
                    "message_id": str(message_id),
                    "telegram_response": response_data
                }
            else:
                error_msg = response_data.get("description", "Error desconocido")
                logger.error(f"Error al enviar mensaje por Telegram: {error_msg}")
                logger.error(f"Response completa: {response_data}")
                return {
                    "status": "error",
                    "message": error_msg,
                    "details": response_data
                }
                
        except Exception as e:
            logger.error(f"Error general al enviar mensaje con Telegram: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    async def send_typing_action(self, chat_id: str) -> bool:
        """Envía la acción de 'escribiendo' para mejorar UX."""
        try:
            payload = {
                "chat_id": chat_id,
                "action": "typing"
            }
            
            response = requests.post(
                f"{self.api_url}/sendChatAction",
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Error al enviar typing action: {str(e)}")
            return False
    
    async def get_webhook_info(self) -> Dict[str, Any]:
        """Obtiene información sobre el webhook configurado."""
        try:
            response = requests.get(f"{self.api_url}/getWebhookInfo")
            return response.json()
        except Exception as e:
            logger.error(f"Error al obtener info del webhook: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def set_webhook(self, webhook_url: str) -> Dict[str, Any]:
        """Configura el webhook para el bot."""
        try:
            payload = {
                "url": webhook_url
            }
            
            # Agregar secret token si está configurado
            if self.webhook_secret:
                payload["secret_token"] = self.webhook_secret
            
            response = requests.post(
                f"{self.api_url}/setWebhook",
                json=payload
            )
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error al configurar webhook: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def delete_webhook(self) -> Dict[str, Any]:
        """Elimina el webhook configurado."""
        try:
            response = requests.post(f"{self.api_url}/deleteWebhook")
            return response.json()
        except Exception as e:
            logger.error(f"Error al eliminar webhook: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def get_me(self) -> Dict[str, Any]:
        """Obtiene información básica del bot."""
        try:
            response = requests.get(f"{self.api_url}/getMe")
            return response.json()
        except Exception as e:
            logger.error(f"Error al obtener información del bot: {str(e)}")
            return {"ok": False, "error": str(e)}
