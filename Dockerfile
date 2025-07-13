FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requisitos
COPY requirements-prod.txt .

# Copiar archivo principal a la raíz
COPY main.py .
COPY rag_system.py .

# Copiar scripts esenciales
COPY scripts/__init__.py scripts/
COPY scripts/run_rag.py scripts/
COPY scripts/gcs_storage.py scripts/
COPY scripts/create_embeddings.py scripts/
COPY scripts/preprocess.py scripts/


# Copiar directorios de módulos
COPY config/ config/
COPY services/ services/
COPY models/ models/
COPY storage/ storage/
COPY utils/ utils/
COPY handlers/ handlers/

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements-prod.txt

# Variables de entorno por defecto
ENV HOST=0.0.0.0
ENV ENVIRONMENT=production

# Exponer puerto
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]