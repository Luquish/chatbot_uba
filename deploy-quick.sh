#!/bin/bash

# =============================================================================
# DEPLOY RÁPIDO - CHATBOT UBA
# =============================================================================
# Script simple para deploy rápido sin verificaciones adicionales

echo "🚀 Deploy rápido del Chatbot UBA..."

gcloud run deploy uba-chatbot \
    --source . \
    --allow-unauthenticated \
    --service-account=chatbot-serviceaccount@drcecim-465823.iam.gserviceaccount.com \
    --region=southamerica-east1 \
    --project=drcecim-465823 \
    --memory=2Gi \
    --cpu=1 \
    --max-instances=10 \
    --min-instances=0 \
    --port=8080 \
    --timeout=300 \
    --concurrency=80 \
    --set-env-vars=ENVIRONMENT=production

echo "✅ Deploy completado!"
echo "🌐 URL: https://uba-chatbot-xxxxx-southamerica-east1.run.app"
echo "📱 Webhook: https://uba-chatbot-xxxxx-southamerica-east1.run.app/webhook/whatsapp" 