#!/bin/bash

# =============================================================================
# SCRIPT DE DEPLOY PARA CHATBOT UBA - CLOUD RUN
# =============================================================================
# Este script despliega el chatbot UBA a Google Cloud Run
# Autor: Sistema de IA
# Fecha: $(date +%Y-%m-%d)

set -e  # Salir si hay algún error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuración
PROJECT_ID="drcecim-465823"
SERVICE_NAME="uba-chatbot"
REGION="southamerica-east1"
SERVICE_ACCOUNT="chatbot-serviceaccount@drcecim-465823.iam.gserviceaccount.com"

print_status "🚀 Iniciando deploy del Chatbot UBA a Cloud Run"
print_status "Proyecto: $PROJECT_ID"
print_status "Servicio: $SERVICE_NAME"
print_status "Región: $REGION"
print_status "Service Account: $SERVICE_ACCOUNT"

# Verificar que estamos en el directorio correcto
if [ ! -f "main.py" ]; then
    print_error "❌ No se encontró main.py. Asegúrate de estar en el directorio del proyecto."
    exit 1
fi

# Verificar que gcloud está instalado
if ! command -v gcloud &> /dev/null; then
    print_error "❌ gcloud CLI no está instalado. Instálalo desde: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Verificar que estamos autenticados
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_warning "⚠️ No hay sesión activa de gcloud. Iniciando autenticación..."
    gcloud auth login
fi

# Verificar que el proyecto está configurado
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    print_status "Configurando proyecto: $PROJECT_ID"
    gcloud config set project $PROJECT_ID
fi

# Verificar que estamos en la región correcta
CURRENT_REGION=$(gcloud config get-value run/region 2>/dev/null || echo "")
if [ "$CURRENT_REGION" != "$REGION" ]; then
    print_status "Configurando región: $REGION"
    gcloud config set run/region $REGION
fi

print_status "📦 Construyendo y desplegando el servicio..."

# Comando de deploy
gcloud run deploy $SERVICE_NAME \
    --source . \
    --allow-unauthenticated \
    --service-account=$SERVICE_ACCOUNT \
    --region=$REGION \
    --project=$PROJECT_ID \
    --memory=2Gi \
    --cpu=1 \
    --max-instances=10 \
    --min-instances=0 \
    --port=8080 \
    --timeout=300 \
    --concurrency=80 \
    --set-env-vars=ENVIRONMENT=production

if [ $? -eq 0 ]; then
    print_success "✅ Deploy completado exitosamente!"
    
    # Obtener la URL del servicio
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
    
    print_success "🌐 URL del servicio: $SERVICE_URL"
    print_success "📱 Webhook URL para WhatsApp: $SERVICE_URL/webhook/whatsapp"
    
    echo ""
    print_status "📋 Próximos pasos:"
    echo "1. Actualiza el webhook en Meta Developer Console:"
    echo "   URL: $SERVICE_URL/webhook/whatsapp"
    echo "   Verify Token: g-A-eAvDRi2f2LOSjLoNq-tcpL2uwxCjpqsWtr1B7uw"
    echo ""
    echo "2. Prueba el health check:"
    echo "   curl $SERVICE_URL/health"
    echo ""
    echo "3. Ejecuta los tests:"
    echo "   python test.py"
    
else
    print_error "❌ Error durante el deploy"
    exit 1
fi

print_success "🎉 ¡Chatbot UBA desplegado y listo para usar!" 