#!/usr/bin/env python
"""
Automatic setup script for the UBA Medicine Chatbot.
This script handles the initial configuration and setup of the project.

Key features:
- Environment variable configuration
- Directory structure creation
- Model download and setup
- Dependencies installation
- WhatsApp API configuration
"""

import os
import sys
import logging
import asyncio
import requests
import signal
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de credenciales de WhatsApp desde variables de entorno
WHATSAPP_API_TOKEN = os.getenv('WHATSAPP_API_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID')
WHATSAPP_WEBHOOK_VERIFY_TOKEN = os.getenv('WHATSAPP_WEBHOOK_VERIFY_TOKEN')

# Variables globales para manejo de procesos
backend_process = None
ngrok_process = None
ngrok_url = None
server_started = False

class AutoSetup:
    """
    Handles automatic setup of the chatbot project.
    
    Responsibilities:
    - Creating necessary directories
    - Setting up environment variables
    - Installing dependencies
    - Downloading and configuring models
    - Setting up WhatsApp integration
    """
    
    def __init__(self):
        """Initialize the AutoSetup class."""
        self.root_dir = Path(__file__).parent.parent
        self.env_file = self.root_dir / '.env'
        self.requirements_file = self.root_dir / 'requirements.txt'
        
    def create_directories(self) -> None:
        """Create necessary project directories."""
        directories = [
            'data',
            'data/raw',
            'data/processed',
            'data/embeddings',
            'models',
            'logs'
        ]
        
        for directory in directories:
            dir_path = self.root_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            
    def setup_env_variables(self) -> None:
        """Configure environment variables."""
        if not self.env_file.exists():
            # Create .env file with default values
            env_vars = {
                'ENVIRONMENT': 'development',
                'MODEL_PATH': 'models/finetuned_model',
                'EMBEDDINGS_DIR': 'data/embeddings',
                'WHATSAPP_API_TOKEN': '',
                'WHATSAPP_PHONE_NUMBER_ID': '',
                'WHATSAPP_BUSINESS_ACCOUNT_ID': '',
                'WHATSAPP_WEBHOOK_VERIFY_TOKEN': '',
                'MY_PHONE_NUMBER': ''
            }
            
            with open(self.env_file, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
                    
            logger.info("Created .env file with default values")
        else:
            logger.info(".env file already exists")
            
    def install_dependencies(self) -> None:
        """Install project dependencies."""
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)], check=True)
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            sys.exit(1)
            
    def download_models(self) -> None:
        """Download and configure required models."""
        try:
            # Download base model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = "microsoft/phi-2"
            model_path = self.root_dir / 'models' / 'base_model'
            
            if not model_path.exists():
                logger.info(f"Downloading base model: {model_name}")
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                logger.info("Base model downloaded successfully")
            else:
                logger.info("Base model already exists")
                
        except Exception as e:
            logger.error(f"Error downloading models: {e}")
            sys.exit(1)
            
    def setup_whatsapp(self) -> None:
        """Configure WhatsApp Business API integration."""
        load_dotenv()
        
        # Check if WhatsApp credentials are configured
        required_vars = [
            'WHATSAPP_API_TOKEN',
            'WHATSAPP_PHONE_NUMBER_ID',
            'WHATSAPP_BUSINESS_ACCOUNT_ID',
            'WHATSAPP_WEBHOOK_VERIFY_TOKEN'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning("Missing WhatsApp credentials:")
            for var in missing_vars:
                logger.warning(f"- {var}")
            logger.info("Please configure these variables in the .env file")
        else:
            logger.info("WhatsApp credentials configured")
            
    def run(self) -> None:
        """Execute the complete setup process."""
        logger.info("Starting automatic setup...")
        
        self.create_directories()
        self.setup_env_variables()
        self.install_dependencies()
        self.download_models()
        self.setup_whatsapp()
        
        logger.info("Setup completed successfully")
        
async def run_process(cmd, name):
    """
    Ejecuta un proceso del sistema de manera asíncrona.
    
    Args:
        cmd (str): Comando a ejecutar
        name (str): Nombre descriptivo del proceso
        
    Returns:
        Process: Objeto de proceso asíncrono
    """
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    logger.info(f"Iniciando {name}...")
    
    return process

async def start_backend():
    """Inicia el backend de Python en segundo plano."""
    global backend_process
    
    # Verificar si el backend ya está en ejecución
    try:
        response = requests.get("http://localhost:8000/health", timeout=1)
        if response.status_code == 200:
            logger.info("Backend ya está en ejecución")
            return True
    except requests.exceptions.RequestException:
        pass
    
    logger.info("Iniciando backend...")
    backend_process = await run_process("python scripts/deploy_backend.py", "backend")
    
    # Esperar a que el servidor esté listo
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                logger.info("Backend iniciado correctamente")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info(f"Esperando a que el backend esté listo ({i+1}/{max_retries})...")
        await asyncio.sleep(1)
    
    logger.error("No se pudo iniciar el backend")
    return False

async def start_ngrok():
    """Inicia ngrok y obtiene la URL pública."""
    global ngrok_process, ngrok_url
    
    # Verificar si ngrok ya está en ejecución
    try:
        response = requests.get("http://localhost:4040/api/tunnels", timeout=1)
        if response.status_code == 200:
            data = response.json()
            if data.get("tunnels"):
                ngrok_url = data["tunnels"][0]["public_url"]
                logger.info(f"ngrok ya está en ejecución en: {ngrok_url}")
                return ngrok_url
    except requests.exceptions.RequestException:
        pass
    
    logger.info("Iniciando ngrok...")
    ngrok_process = await run_process("ngrok http 8000", "ngrok")
    
    # Esperar a que ngrok esté listo
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:4040/api/tunnels", timeout=1)
            if response.status_code == 200:
                data = response.json()
                if data.get("tunnels"):
                    ngrok_url = data["tunnels"][0]["public_url"]
                    logger.info(f"ngrok iniciado en: {ngrok_url}")
                    return ngrok_url
        except requests.exceptions.RequestException:
            pass
        
        logger.info(f"Esperando a que ngrok esté listo ({i+1}/{max_retries})...")
        await asyncio.sleep(1)
    
    logger.error("No se pudo iniciar ngrok")
    return None

async def verify_token():
    """Verifica si el token de WhatsApp es válido y cuándo expira."""
    try:
        # Verificar token
        url = f"https://graph.facebook.com/debug_token?input_token={WHATSAPP_API_TOKEN}&access_token={WHATSAPP_API_TOKEN}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            token_data = data.get("data", {})
            
            # Verificar si el token es válido
            if token_data.get("is_valid", False):
                expires_at = token_data.get("expires_at")
                app_id = token_data.get("app_id")
                
                if expires_at:
                    # Convertir timestamp a fecha
                    expiry_date = datetime.fromtimestamp(expires_at)
                    now = datetime.now()
                    days_left = (expiry_date - now).days
                    
                    if days_left < 1:
                        hours_left = (expiry_date - now).seconds // 3600
                        logger.warning(f"⚠️ El token expirará en {hours_left} horas")
                    else:
                        logger.info(f"Token válido. Expira en {days_left} días")
                        
                    return True
                else:
                    logger.info("Token válido sin fecha de expiración")
                    return True
            else:
                error = token_data.get("error", {}).get("message", "Error desconocido")
                logger.error(f"Token inválido: {error}")
                return False
        else:
            logger.error(f"Error al verificar token: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error al verificar token: {str(e)}")
        return False

async def send_test_message():
    """Envía un mensaje de prueba."""
    try:
        response = requests.get("http://localhost:8000/test-message")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                logger.info(f"Mensaje de prueba enviado correctamente: {data.get('message_id')}")
                return True
            else:
                logger.error(f"Error al enviar mensaje de prueba: {data.get('error')}")
                return False
        else:
            logger.error(f"Error al enviar mensaje de prueba: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error al enviar mensaje de prueba: {str(e)}")
        return False

async def cleanup(signum=None, frame=None):
    """Limpia todos los procesos al terminar."""
    logger.info("Limpiando procesos...")
    
    if backend_process:
        backend_process.terminate()
        logger.info("Backend detenido")
    
    if ngrok_process:
        ngrok_process.terminate()
        logger.info("ngrok detenido")
    
    # No usamos sys.exit aquí para permitir una salida limpia
    return

def sync_cleanup(signum=None, frame=None):
    """Versión sincrónica del cleanup para el manejador de señales."""
    if server_started:
        # Si el servidor ya está iniciado, cerramos todo
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cleanup())
    sys.exit(0)

async def main():
    """Función principal."""
    global server_started
    
    # Registrar manejador de señales (usando la versión sincrónica)
    signal.signal(signal.SIGINT, sync_cleanup)
    signal.signal(signal.SIGTERM, sync_cleanup)
    
    try:
        # Verificar variables de entorno
        if not all([
            WHATSAPP_API_TOKEN,
            WHATSAPP_PHONE_NUMBER_ID,
            WHATSAPP_BUSINESS_ACCOUNT_ID,
            WHATSAPP_WEBHOOK_VERIFY_TOKEN
        ]):
            logger.error("Faltan variables de entorno obligatorias en .env")
            return
        
        # Verificar si el token es válido
        if not await verify_token():
            logger.error("El token de WhatsApp no es válido. Por favor actualícelo en .env")
            return
        
        # Iniciar backend
        if not await start_backend():
            logger.error("No se pudo iniciar el backend")
            return
        
        # Iniciar ngrok
        ngrok_url = await start_ngrok()
        if not ngrok_url:
            logger.error("No se pudo iniciar ngrok")
            return
        
        # Instrucciones para configuración manual
        logger.info("\n====== CONFIGURACIÓN COMPLETADA ======")
        logger.info(f"📱 Backend: http://localhost:8000")
        logger.info(f"🔄 ngrok: {ngrok_url}")
        
        logger.info("\n📋 INSTRUCCIONES PARA CONFIGURACIÓN MANUAL:")
        logger.info("1. Copie esta URL para Glitch:")
        logger.info(f"   {ngrok_url}/api/whatsapp/message")
        logger.info("2. Actualice la variable BACKEND_URL en Glitch con la URL anterior")
        logger.info("3. Asegúrese de que WEBHOOK_VERIFY_TOKEN en Glitch coincida con su .env local")
        logger.info("4. Asegúrese de que GRAPH_API_TOKEN en Glitch tenga el mismo valor que WHATSAPP_API_TOKEN en su .env local")
        
        logger.info("\n🌐 CONFIGURACIÓN DE WEBHOOK EN META:")
        logger.info("URL del Webhook: https://su-proyecto-glitch.glitch.me/webhook")
        logger.info(f"Token de verificación: {WHATSAPP_WEBHOOK_VERIFY_TOKEN}")
        
        logger.info("\nMantenga este script en ejecución para desarrollo.")
        logger.info("Presione Ctrl+C para detener todos los servicios.\n")
        
        server_started = True
        
        # Opcional: Enviar mensaje de prueba
        await send_test_message()
        
        # Mantener el script en ejecución
        while True:
            await asyncio.sleep(300)  # Verificar cada 5 minutos
            await verify_token()
            
    except KeyboardInterrupt:
        logger.info("\nDeteniendo servicios...")
        await cleanup()  # Asegurarse de llamar cleanup en caso de KeyboardInterrupt
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
    finally:
        # Asegurarse de que cleanup se llame al salir
        await cleanup()

def main():
    """Main function to run the setup."""
    setup = AutoSetup()
    setup.run()
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Este bloque se ejecutará si se presiona Ctrl+C antes de que se inicie server_started
        if not server_started:
            # Usar un EventLoop temporal para llamar a cleanup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(cleanup()) 