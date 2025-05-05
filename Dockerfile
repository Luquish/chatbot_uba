FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo los archivos necesarios para producción
COPY requirements-prod.txt .
COPY scripts/deploy_backend.py scripts/
COPY scripts/run_rag.py scripts/
COPY scripts/calendar_service.py scripts/
COPY scripts/date_utils.py scripts/
COPY scripts/gcs_storage.py scripts/
COPY scripts/__init__.py scripts/
COPY config/calendar_config.py config/

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements-prod.txt

# Variables de entorno por defecto
ENV HOST=0.0.0.0
ENV ENVIRONMENT=production

# Exponer puerto
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "scripts/deploy_backend.py"] 