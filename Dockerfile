# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools required for compiling Python packages
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final image ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY main.py .
COPY rag_system.py .

# Essential scripts (only GCS storage and RAG runner)
COPY scripts/__init__.py scripts/
COPY scripts/run_rag.py scripts/
COPY scripts/gcs_storage.py scripts/

# Module directories
COPY config/ config/
COPY services/ services/
COPY models/ models/
COPY storage/ storage/
COPY utils/ utils/
COPY handlers/ handlers/

# Default environment variables
ENV HOST=0.0.0.0
ENV ENVIRONMENT=production

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
