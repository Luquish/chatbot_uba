#!/usr/bin/env python
"""
Script para automatizar la configuración del entorno de desarrollo para WhatsApp.
Enfoque manual - Este script:
1. Inicia el backend de Python en segundo plano
2. Inicia ngrok y obtiene la URL pública
3. Verifica el token de WhatsApp
4. Proporciona instrucciones para configurar manualmente Glitch y Meta
"""

import os
import sys
import logging
import asyncio
import requests
import signal
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración
WHATSAPP_API_TOKEN = os.getenv('WHATSAPP_API_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv('WHATSAPP_BUSINESS_ACCOUNT_ID')
WHATSAPP_WEBHOOK_VERIFY_TOKEN = os.getenv('WHATSAPP_WEBHOOK_VERIFY_TOKEN')

# Variables globales para procesos
backend_process = None
ngrok_process = None
ngrok_url = None
server_started = False

async def run_process(cmd, name):
    """Ejecuta un proceso e imprime su salida."""
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
                logger.info("Backend iniciado...")
                return True
        except requests.exceptions.RequestException:
            pass
        
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
                    logger.info(f"ngrok iniciado...")
                    return ngrok_url
        except requests.exceptions.RequestException:
            pass
        
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
        my_phone = os.getenv('MY_PHONE_NUMBER')
        if not my_phone:
            logger.error("No se ha configurado MY_PHONE_NUMBER en el archivo .env")
            return False
            
        logger.info(f"Enviando mensaje de prueba al número: {my_phone}")
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