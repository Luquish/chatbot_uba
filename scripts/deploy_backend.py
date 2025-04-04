"""
Main backend for the UBA Medicina chatbot.
Implements the REST API and WhatsApp Business API integration.

This module is responsible for:
1. Handling incoming WhatsApp webhooks
2. Processing and validating messages
3. Integrating with the RAG system for response generation
4. Sending messages through the WhatsApp API
5. Managing authentication and security

Key features:
- Webhook signature validation
- Phone number normalization
- Robust error handling
- Detailed logging
- CORS and security
"""

import os
import logging
import json
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from run_rag import RAGSystem
import uvicorn
import re

# Load environment variables
load_dotenv()

# Detailed logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to see all messages
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine environment (development or production)
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
logger.info(f"Starting in environment: {ENVIRONMENT}")

# WhatsApp Business API configuration from environment variables
WHATSAPP_API_TOKEN = os.getenv('WHATSAPP_API_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID')
WHATSAPP_WEBHOOK_VERIFY_TOKEN = os.getenv('WHATSAPP_WEBHOOK_VERIFY_TOKEN')

# Test phone number
MY_PHONE_NUMBER = os.getenv('MY_PHONE_NUMBER')

class WhatsAppHandler:
    """
    Handles interaction with the WhatsApp Business API.
    
    Responsibilities:
    - Message sending and receiving
    - Webhook validation
    - Phone number normalization
    - Error handling and retries
    """
    
    def __init__(self, api_token: str, phone_number_id: str, business_account_id: str):
        """
        Initializes the WhatsApp handler with necessary credentials.
        
        Args:
            api_token (str): API access token
            phone_number_id (str): Phone number ID
            business_account_id (str): Business account ID
        """
        self.api_token = api_token
        self.phone_number_id = phone_number_id
        self.business_account_id = business_account_id
        self.api_url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.webhook_verify_token = os.getenv('WHATSAPP_WEBHOOK_VERIFY_TOKEN')
    
    def normalize_phone_number(self, phone: str) -> str:
        """Normalizes phone number for Argentina, removing '9' if it exists."""
        # Remove any non-numeric characters
        clean_number = re.sub(r'[^0-9]', '', phone)
        
        # If it's an Argentine number (starts with 54) and has 9 after country code
        if clean_number.startswith('54') and len(clean_number) > 2:
            if clean_number[2] == '9':
                # Remove 9 after country code
                clean_number = clean_number[:2] + clean_number[3:]
                logger.info(f"Number normalized from {phone} to {clean_number}")
        
        return clean_number
    
    async def validate_request(self, request: Request) -> bool:
        # Verify webhook verification token
        if request.method == "GET":
            mode = request.query_params.get("hub.mode")
            token = request.query_params.get("hub.verify_token")
            challenge = request.query_params.get("hub.challenge")
            
            if mode == "subscribe" and token == self.webhook_verify_token:
                return True
            return False
            
        # Validate webhook signature for POST
        if ENVIRONMENT == "development":
            logger.warning("Skipping webhook signature validation in development")
            return True
            
        # Get signature from header
        signature = request.headers.get("x-hub-signature-256")
        if not signature:
            logger.error("No signature found in header")
            return False
            
        # Get request body
        body = await request.body()
        
        # Calculate HMAC SHA256
        import hmac
        import hashlib
        expected_signature = hmac.new(
            self.webhook_verify_token.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
    
    async def parse_message(self, request: Request) -> Dict[str, str]:
        try:
            data = await request.json()
            logger.debug(f"Data received from WhatsApp API: {data}")
            
            # Standard WhatsApp API webhook structure
            entry = data.get('entry', [])[0]
            changes = entry.get('changes', [])[0]
            value = changes.get('value', {})
            message = value.get('messages', [])[0]
            
            # Extract message information
            message_body = message.get('text', {}).get('body', '')
            from_number = message.get('from', '')
            message_id = message.get('id', '')
            timestamp = message.get('timestamp', '')
            
            # Extract contact information if available
            contacts = value.get('contacts', [])
            contact_info = contacts[0] if contacts else {}
            profile_name = contact_info.get('profile', {}).get('name', '')
            
            return {
                'body': message_body,
                'from': from_number,
                'to': self.phone_number_id,
                'message_id': message_id,
                'timestamp': timestamp,
                'profile_name': profile_name
            }
        except (IndexError, KeyError) as e:
            logger.error(f"Error parsing WhatsApp API message: {str(e)}")
            logger.error(f"Message structure: {data}")
            return {'body': '', 'from': '', 'to': ''}
        except Exception as e:
            logger.error(f"General error parsing WhatsApp API message: {str(e)}")
            logger.error(f"Request headers: {request.headers}")
            logger.error(f"Request method: {request.method}")
            return {'body': '', 'from': '', 'to': ''}
    
    async def send_message(self, to: str, body: str) -> Dict[str, Any]:
        try:
            # Normalize phone number
            formatted_phone = self.normalize_phone_number(to)
            
            logger.info(f"Sending message to {formatted_phone}: '{body}'")
            
            # Prepare payload according to official documentation
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": formatted_phone,
                "type": "text",
                "text": {
                    "preview_url": False,  # Disable URL preview
                    "body": body
                }
            }
            
            # Show debug information
            logger.debug(f"URL: {self.api_url}")
            logger.debug(f"Headers: {json.dumps(self.headers, indent=2)}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Send message
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            # Process response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON response. Status code: {response.status_code}, Text: {response.text}")
                return {
                    'status': 'error',
                    'message': f"Non-JSON response error: {response.text}"
                }
            
            logger.debug(f"API Response: {json.dumps(response_data, indent=2)}")
            
            if response.status_code == 200:
                message_id = response_data.get('messages', [{}])[0].get('id', '')
                logger.info(f"Message sent successfully. ID: {message_id}")
                return {
                    'status': 'success',
                    'message_id': message_id,
                    'whatsapp_api_response': response_data
                }
            else:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                error_code = response_data.get('error', {}).get('code', '')
                error_type = response_data.get('error', {}).get('type', '')
                error_fbtrace_id = response_data.get('error', {}).get('fbtrace_id', '')
                
                logger.error(f"Error sending message via WhatsApp API: {error_message}")
                logger.error(f"Code: {error_code}, Type: {error_type}, FB Trace ID: {error_fbtrace_id}")
                
                return {
                    'status': 'error',
                    'message': error_message,
                    'code': error_code,
                    'type': error_type,
                    'details': response_data
                }
        except Exception as e:
            logger.error(f"General error sending message with WhatsApp API: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

# Function to get WhatsApp handler
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
            detail="WhatsApp integration not available. Missing credentials."
        )

# Initialize FastAPI
app = FastAPI(
    title="UBA Chatbot - API",
    description="API for the UBA Faculty of Medicine educational chatbot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system (use environment variables)
model_path = os.getenv('MODEL_PATH', 'models/finetuned_model')
embeddings_dir = os.getenv('EMBEDDINGS_DIR', 'data/embeddings')

# Initialize RAG with error handling
try:
    logger.info(f"Initializing RAG system with model_path={model_path}, embeddings_dir={embeddings_dir}")
    rag_system = RAGSystem(model_path, embeddings_dir)
    rag_initialized = True
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG: {str(e)}", exc_info=True)
    logger.warning("Using alternative response system")
    rag_initialized = False
    
    # Create simple alternative function for development
    class FallbackRAG:
        def process_query(self, query):
            logger.info(f"Using fallback system for query: {query}")
            return {
                "response": f"I received your message: '{query}'. The RAG system is under maintenance, but we can have a basic conversation."
            }
    
    rag_system = FallbackRAG()

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Webhook for receiving WhatsApp messages.
    
    Args:
        request (Request): FastAPI request
        
    Returns:
        Dict: Response with processing status
    """
    try:
        # For debugging, log method and headers
        logger.info(f"Webhook received - Method: {request.method}")
        logger.info(f"Headers: {request.headers}")
        
        # Get request body
        body = await request.body()
        
        # Validate webhook signature in production
        if ENVIRONMENT == "production":
            signature = request.headers.get("x-hub-signature-256", "")
            if not signature:
                logger.error("Signature not found in headers")
                raise HTTPException(status_code=403, detail="Signature not found")
                
            # Calculate and verify signature
            import hmac
            import hashlib
            
            expected_signature = hmac.new(
                WHATSAPP_WEBHOOK_VERIFY_TOKEN.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(f"sha256={expected_signature}", signature):
                logger.error("Invalid signature")
                raise HTTPException(status_code=403, detail="Invalid signature")
        
        # Parse JSON body
        try:
            data = await request.json()
            logger.debug(f"Received data: {json.dumps(data, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        # Check if it's a message
        if "object" not in data:
            logger.warning("Object not found in request")
            return {"status": "ignored"}
            
        if data["object"] != "whatsapp_business_account":
            logger.warning(f"Unexpected object: {data['object']}")
            return {"status": "ignored"}
            
        # Process entry
        try:
            entry = data["entry"][0]
            changes = entry["changes"][0]
            value = changes["value"]
            
            # If we have a message or any event, forward to backend
            if "messages" in value:
                message = value.get("messages", [])[0]
                message_body = message.get("text", {}).get("body", "")
                from_number = message.get("from", "")
                message_id = message.get("id", "")
                logger.info(f"New message received - ID: {message_id}, From: {from_number}, Content: '{message_body}'")
                
                # Normalize sender number
                whatsapp_handler = get_whatsapp_handler()
                from_number = whatsapp_handler.normalize_phone_number(from_number)
                
                # Generate response using RAG
                try:
                    result = rag_system.process_query(message_body)
                    response_text = result["response"]
                except Exception as e:
                    logger.error(f"Error processing RAG query: {str(e)}")
                    response_text = "I'm sorry, I had a problem processing your message. Please try again."
                
                # Send response
                send_result = await whatsapp_handler.send_message(
                    from_number,
                    response_text
                )
                
                return {
                    "status": "success",
                    "message": "Response sent",
                    "details": send_result
                }
            elif "statuses" in value:
                status = value.get("statuses", [])[0]
                status_type = status.get("status")
                message_id = status.get("id")
                recipient = status.get("recipient_id")
                
                if status_type == "sent":
                    logger.debug(f"✓ Message {message_id} sent to {recipient}")
                elif status_type == "delivered":
                    logger.debug(f"✓✓ Message {message_id} delivered to {recipient}")
                elif status_type == "read":
                    logger.debug(f"✓✓✓ Message {message_id} read by {recipient}")
                else:
                    logger.debug(f"Unknown status for message {message_id}: {status_type}")
                
                return {"status": "success", "message": f"Status {status_type} processed"}
            else:
                logger.warning("Webhook received without messages or statuses")
                return {"status": "ignored", "message": "Webhook without processable content"}
                
        except (KeyError, IndexError) as e:
            logger.error(f"Error processing entry: {str(e)}")
            return {"status": "error", "message": "Invalid entry format"}
        
        return {"status": "ignored", "message": "Non-processable event"}
            
    except Exception as e:
        logger.error(f"Error in WhatsApp webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    """
    Endpoint for verifying WhatsApp webhook.
    """
    try:
        # Get query parameters
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        
        # Log for debugging
        logger.info(f"Webhook verification received:")
        logger.info(f"Mode: {mode}")
        logger.info(f"Received token: {token}")
        logger.info(f"Expected token: {WHATSAPP_WEBHOOK_VERIFY_TOKEN}")
        logger.info(f"Challenge: {challenge}")
        
        # Verify mode and token
        if mode and token and mode == "subscribe" and token == WHATSAPP_WEBHOOK_VERIFY_TOKEN:
            if not challenge:
                logger.error("Challenge not found")
                return {"status": "error", "message": "No challenge found"}
                
            logger.info(f"Verification successful, returning challenge: {challenge}")
            return int(challenge)
            
        logger.error("Verification failed - Invalid token or mode")
        raise HTTPException(status_code=403, detail="Forbidden")
            
    except Exception as e:
        logger.error(f"Error in webhook verification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-message")
async def test_message():
    """
    Sends a test message to the WhatsApp number configured in MY_PHONE_NUMBER.
    
    Returns:
        dict: Result of sending the test message
    """
    if not MY_PHONE_NUMBER:
        return {"error": "MY_PHONE_NUMBER not configured in environment variables"}
    
    try:
        # Verify credentials
        if not WHATSAPP_API_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            return {"error": "Missing WhatsApp Business API credentials in environment variables"}
        
        # Format phone number according to WhatsApp API format
        # (without '+' sign, only digits)
        formatted_phone = MY_PHONE_NUMBER
        # Remove '+' if exists and any other non-numeric character
        formatted_phone = re.sub(r'[^0-9]', '', formatted_phone)
            
        logger.info(f"Sending test message to: {formatted_phone}")
        
        # Configure headers for WhatsApp API
        headers = {
            "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Prepare message data
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": formatted_phone,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": "Hello! This is a test message from the UBA Medicine Chatbot. If you receive this, the configuration is correct."
            }
        }
        
        # Send request to WhatsApp API
        response = requests.post(
            f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=data
        )
        
        # Verify response
        response_data = response.json()
        
        if response.status_code == 200:
            return {
                "success": True,
                "message_id": response_data.get("messages", [{}])[0].get("id"),
                "status": "sent",
                "whatsapp_api_response": response_data
            }
        else:
            error_message = response_data.get('error', {}).get('message', 'Unknown error')
            error_code = response_data.get('error', {}).get('code', '')
            return {
                "success": False,
                "error": f"Error sending message: {error_message} (Code: {error_code})",
                "details": response_data
            }
    
    except Exception as e:
        logging.error(f"Error sending test message: {str(e)}")
        return {"error": f"Error sending test message: {str(e)}"}

@app.get("/health")
async def health_check():
    """Endpoint to verify service status."""
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
        "model_path": model_path,
        "embeddings_dir": embeddings_dir
    }
    
    # Add WhatsApp configuration information
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
    Endpoint to test the chatbot without WhatsApp.
    
    Args:
        message (Dict[str, str]): Dictionary with the message
        
    Returns:
        Dict: Chatbot response
    """
    try:
        query = message.get("message")
        if not query:
            raise HTTPException(status_code=400, detail="Message not provided")
            
        result = rag_system.process_query(query)
        return result
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/whatsapp/message")
async def receive_whatsapp_message(request: Request):
    """
    Endpoint for receiving WhatsApp messages redirected from Glitch.
    
    This endpoint is called by the Glitch webhook when it receives a message.
    """
    # Add entry log to confirm endpoint is being called
    logger.info("======= MESSAGE RECEIVED IN /api/whatsapp/message =======")
    
    try:
        # Log headers and query params for debugging
        logger.debug(f"Headers: {dict(request.headers)}")
        
        # Get message data as text for debugging
        body_bytes = await request.body()
        body_text = body_bytes.decode('utf-8')
        logger.debug(f"RAW Body: {body_text}")
        
        # Get message data
        try:
            data = await request.json()
            logger.debug(f"JSON data received: {json.dumps(data, indent=2)}")
        except Exception as e:
            logger.error(f"Error reading request body: {str(e)}")
            return {"status": "error", "message": f"JSON error: {str(e)}", "body": body_text}
        
        # Try to process message according to complete webhook format or simplified format
        message_body = ""
        from_number = ""
        
        # Check if we're receiving complete webhook or just simplified message
        if "object" in data and data.get("object") == "whatsapp_business_account":
            # Complete webhook format
            logger.debug("Detected complete webhook format")
            try:
                entry = data.get("entry", [])[0]
                changes = entry.get("changes", [])[0]
                value = changes.get("value", {})
                
                # Check if there are messages
                if "messages" in value:
                    message = value.get("messages", [])[0]
                    message_body = message.get("text", {}).get("body", "")
                    from_number = message.get("from", "")
                    
                    # Extract business phone number ID from metadata
                    business_phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
                    
                    logger.info(f"Complete webhook - From: {from_number}, Message: '{message_body}'")
                else:
                    # Could be a status message (read, delivered, etc)
                    logger.debug("Status event received, not a text message")
                    return {"status": "ignored", "message": "Status event, not a message"}
            except (IndexError, KeyError) as e:
                logger.error(f"Error extracting data from complete webhook: {str(e)}")
                return {"status": "error", "message": f"Webhook structure error: {str(e)}"}
        else:
            # Simplified format sent by current server.js
            message = data.get("message", {})
            
            if "text" in message:
                message_body = message.get("text", {}).get("body", "")
                from_number = message.get("from", "")
            
            business_phone_number_id = data.get("business_phone_number_id", "")
            logger.info(f"Simplified format - From: {from_number}, Message: '{message_body}'")
        
        # Log extracted data
        logger.info(f"Extracted message - From: {from_number}, Body: '{message_body}'")
        
        if not message_body or not from_number:
            logger.error("Error: Message without body or sender")
            return {"status": "error", "message": "Incomplete data", "received_data": data}
        
        # Normalize sender number
        whatsapp_handler = get_whatsapp_handler()
        from_number = whatsapp_handler.normalize_phone_number(from_number)
        logger.info(f"Normalized number: {from_number}")
        
        # Process message using RAG
        try:
            logger.info(f"Processing message with RAG: '{message_body}'")
            result = rag_system.process_query(message_body)
            response_text = result["response"]
            logger.info(f"RAG response generated: {response_text}")
        except Exception as e:
            logger.error(f"Error processing with RAG: {str(e)}")
            response_text = "I'm sorry, I had a problem processing your message. Please try again later."
        
        # Send response via WhatsApp
        try:
            send_result = await whatsapp_handler.send_message(
                from_number,
                response_text
            )
            logger.debug(f"Send result: {json.dumps(send_result, indent=2)}")
            
            return {
                "status": "success",
                "message": "Response sent successfully",
                "response_text": response_text,
                "details": send_result
            }
        except Exception as e:
            logger.error(f"Error sending: {str(e)}")
            return {"status": "error", "message": f"Error sending: {str(e)}"}
        
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        import traceback
        return {
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/test-webhook")
async def test_webhook():
    """Endpoint to test direct connection."""
    logger.warning("Test endpoint /test-webhook called")
    
    try:
        # Verify we can send messages
        if MY_PHONE_NUMBER:
            try:
                formatted_phone = re.sub(r'[^0-9]', '', MY_PHONE_NUMBER)
                
                whatsapp_handler = get_whatsapp_handler()
                result = await whatsapp_handler.send_message(
                    formatted_phone,
                    "This is a test message directly from the /test-webhook endpoint."
                )
                
                if result.get('status') == 'success':
                    return {
                        "status": "success",
                        "message": "Test message sent directly",
                        "to": formatted_phone,
                        "details": result
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Error sending test message",
                        "details": result
                    }
            except Exception as e:
                logger.error(f"Error sending direct message: {str(e)}", exc_info=True)
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": "No phone number configured for testing"}
    except Exception as e:
        logger.error(f"Error in test-webhook: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

def main():
    """Main function to run the server."""
    # Server configuration from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Embeddings path: {embeddings_dir}")
    if MY_PHONE_NUMBER:
        logger.info(f"Test number configured: {MY_PHONE_NUMBER}")
    
    # Start server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main() 