# 🚀 Deploy del Chatbot UBA - Cloud Run

## Scripts de Deploy

### 1. Deploy Completo (Recomendado)
```bash
./deploy.sh
```
**Características:**
- ✅ Verificaciones de seguridad
- ✅ Validación de configuración
- ✅ Autenticación automática
- ✅ Configuración de proyecto/región
- ✅ Output con colores y información detallada

### 2. Deploy Rápido
```bash
./deploy-quick.sh
```
**Características:**
- ⚡ Deploy directo sin verificaciones
- ⚡ Para cuando ya tienes todo configurado
- ⚡ Output mínimo

## Configuración Requerida

### Prerequisitos
```bash
# Instalar Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Autenticarse
gcloud auth login

# Configurar proyecto
gcloud config set project drcecim-465823
```

### Variables de Entorno
Las variables están configuradas en Google Secret Manager y se cargan automáticamente en Cloud Run.

## Post-Deploy

### 1. Configurar Webhook de Telegram
Ejecutar comando para configurar webhook:
```bash
curl -X POST "https://uba-chatbot-xxxxx-southamerica-east1.run.app/telegram/setup-webhook" \
     -H "Content-Type: application/json" \
     -d '{"webhook_url": "https://uba-chatbot-xxxxx-southamerica-east1.run.app/webhook/telegram"}'
```

### 2. Probar el Servicio
```bash
# Health check
curl https://uba-chatbot-xxxxx-southamerica-east1.run.app/health

# Test del chatbot
curl -X POST https://uba-chatbot-xxxxx-southamerica-east1.run.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Cómo se cuentan las materias del CBC?"}'
```

### 3. Monitoreo
```bash
# Ver logs
gcloud run services logs read uba-chatbot --region=southamerica-east1

# Ver métricas
gcloud run services describe uba-chatbot --region=southamerica-east1
```

## Configuración de Cloud Run

### Recursos
- **CPU**: 1 vCPU
- **Memoria**: 2GB
- **Instancias**: 0-10 (auto-scaling)
- **Concurrencia**: 80 requests por instancia
- **Timeout**: 300 segundos

### Variables de Entorno
- `ENVIRONMENT=production`
- Todas las demás variables se obtienen de Google Secret Manager

## Troubleshooting

### Error: "Service account not found"
```bash
# Verificar que el service account existe
gcloud iam service-accounts list --filter="email:chatbot-serviceaccount"
```

### Error: "Permission denied"
```bash
# Verificar permisos
gcloud projects get-iam-policy drcecim-465823
```

### Error: "Build failed"
```bash
# Verificar que estás en el directorio correcto
ls main.py rag_system.py

# Verificar Dockerfile
cat Dockerfile
```

## URLs Importantes

- **Servicio**: `https://uba-chatbot-xxxxx-southamerica-east1.run.app`
- **Health Check**: `/health`
- **Webhook Telegram**: `/webhook/telegram`
- **Chat API**: `/chat`
- **Test**: `/test-webhook`

## Comandos Útiles

```bash
# Ver estado del servicio
gcloud run services describe uba-chatbot --region=southamerica-east1

# Ver logs en tiempo real
gcloud run services logs tail uba-chatbot --region=southamerica-east1

# Actualizar variables de entorno
gcloud run services update uba-chatbot --region=southamerica-east1 --set-env-vars=ENVIRONMENT=production

# Escalar manualmente
gcloud run services update uba-chatbot --region=southamerica-east1 --min-instances=1
``` 