# Use Python 3.9 slim as base image for efficiency
FROM python:3.9-slim

# Set maintainer information
LABEL maintainer="Fake News Detection Team"
LABEL description="RoBERTa-based Fake News Detection Model"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create cache directories
RUN mkdir -p /app/.cache/transformers /app/.cache/huggingface

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "app.py"]
