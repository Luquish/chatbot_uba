import os
import logging
import json
import requests
import re
from typing import Dict, Any
from fastapi import Request

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logger = logging.getLogger(__name__)


class WhatsAppHandler:
    def __init__(self, api_token: str, phone_number_id: str, business_account_id: str):
        self.api_token = api_token
        self.phone_number_id = phone_number_id
        self.business_account_id = business_account_id
        self.api_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self.webhook_verify_token = os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN")

    def normalize_phone_number(self, phone: str) -> str:
        """Normaliza el número de teléfono para Argentina, removiendo el '9' si existe."""
        clean_number = re.sub(r"[^0-9]", "", phone)

        if clean_number.startswith("54") and len(clean_number) > 2:
            if clean_number[2] == "9":
                clean_number = clean_number[:2] + clean_number[3:]
                logger.info(f"Número normalizado de {phone} a {clean_number}")

        return clean_number

    async def validate_request(self, request: Request) -> bool:
        if request.method == "GET":
            mode = request.query_params.get("hub.mode")
            token = request.query_params.get("hub.verify_token")
            if mode == "subscribe" and token == self.webhook_verify_token:
                return True
            return False

        if ENVIRONMENT == "development":
            logger.warning("Omitiendo validación de firma de webhook en desarrollo")
            return True

        signature = request.headers.get("x-hub-signature-256")
        if not signature:
            logger.error("No se encontró firma en el header")
            return False

        body = await request.body()

        import hmac
        import hashlib

        expected_signature = hmac.new(
            self.webhook_verify_token.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(f"sha256={expected_signature}", signature)

    async def parse_message(self, request: Request) -> Dict[str, str]:
        try:
            data = await request.json()
            logger.debug(f"Datos recibidos de WhatsApp API: {data}")

            entry = data.get("entry", [])[0]
            changes = entry.get("changes", [])[0]
            value = changes.get("value", {})
            message = value.get("messages", [])[0]

            message_body = message.get("text", {}).get("body", "")
            from_number = message.get("from", "")
            message_id = message.get("id", "")
            timestamp = message.get("timestamp", "")

            contacts = value.get("contacts", [])
            contact_info = contacts[0] if contacts else {}
            profile_name = contact_info.get("profile", {}).get("name", "")

            return {
                "body": message_body,
                "from": from_number,
                "to": self.phone_number_id,
                "message_id": message_id,
                "timestamp": timestamp,
                "profile_name": profile_name,
            }
        except (IndexError, KeyError) as e:
            logger.error(f"Error al parsear mensaje de WhatsApp API: {str(e)}")
            logger.error(f"Estructura del mensaje: {data}")
            return {"body": "", "from": "", "to": ""}
        except Exception as e:
            logger.error(f"Error general al parsear mensaje de WhatsApp API: {str(e)}")
            logger.error(f"Request headers: {request.headers}")
            logger.error(f"Request method: {request.method}")
            return {"body": "", "from": "", "to": ""}

    async def send_message(self, to: str, body: str) -> Dict[str, Any]:
        try:
            formatted_phone = self.normalize_phone_number(to)

            logger.info(f"Enviando mensaje a {formatted_phone}: '{body}'")

            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": formatted_phone,
                "type": "text",
                "text": {
                    "preview_url": False,
                    "body": body,
                },
            }

            logger.debug(f"URL: {self.api_url}")
            logger.debug(f"Headers: {json.dumps(self.headers, indent=2)}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )

            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logger.error(
                    f"Error al decodificar respuesta JSON. Status code: {response.status_code}, Text: {response.text}"
                )
                return {"status": "error", "message": f"Error de respuesta no-JSON: {response.text}"}

            logger.debug(f"Respuesta API: {json.dumps(response_data, indent=2)}")

            if response.status_code == 200:
                message_id = response_data.get("messages", [{}])[0].get("id", "")
                logger.info(f"Mensaje enviado exitosamente. ID: {message_id}")
                return {
                    "status": "success",
                    "message_id": message_id,
                    "whatsapp_api_response": response_data,
                }
            else:
                error_message = response_data.get("error", {}).get("message", "Error desconocido")
                error_code = response_data.get("error", {}).get("code", "")
                error_type = response_data.get("error", {}).get("type", "")
                error_fbtrace_id = response_data.get("error", {}).get("fbtrace_id", "")

                logger.error(f"Error al enviar mensaje por WhatsApp API: {error_message}")
                logger.error(f"Código: {error_code}, Tipo: {error_type}, FB Trace ID: {error_fbtrace_id}")

                return {
                    "status": "error",
                    "message": error_message,
                    "code": error_code,
                    "type": error_type,
                    "details": response_data,
                }
        except Exception as e:
            logger.error(f"Error general al enviar mensaje con WhatsApp API: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}

