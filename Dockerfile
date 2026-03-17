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
COPY main.py .
COPY static/ ./static/
COPY app/ ./app/

ENV PORT=8080
EXPOSE ${PORT}

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 1800
