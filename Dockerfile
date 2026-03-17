FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Tesseract required by img2table)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app.py .
COPY app/ ./app/

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/_stcore/health')" || exit 1

CMD streamlit run streamlit_app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableXsrfProtection=false \
    --server.enableCORS=false \
    --server.maxUploadSize=500 \
    --server.maxMessageSize=500
