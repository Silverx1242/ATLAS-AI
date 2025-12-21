FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /chroma

# Instalar ChromaDB
RUN pip install --no-cache-dir \
    chromadb \
    fastapi \
    uvicorn

# Crear directorio para datos persistentes
RUN mkdir -p /chroma/data

# Variables de entorno
ENV CHROMA_DB_IMPL=duckdb+parquet \
    CHROMA_SERVER_HOST=0.0.0.0 \
    CHROMA_SERVER_PORT=8000 \
    CHROMA_SERVER_CORS_ORIGINS="*" \
    PERSIST_DIRECTORY=/chroma/data

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar el servidor
CMD ["uvicorn", "chromadb.app:app", "--host", "0.0.0.0", "--port", "8000"]
