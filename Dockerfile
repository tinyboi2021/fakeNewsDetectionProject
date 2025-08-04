FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
LABEL maintainer="Fake News Detection Team"

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-setuptools python3-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 3. COPY ONLY requirements.txt first for cache efficiency
COPY requirements.txt ./

# 4. Install PyTorch with CUDA from the official wheel
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 5. Install all the other dependencies (they *should NOT* include torch)
RUN pip3 install --no-cache-dir -r requirements.txt

# 6. COPY the rest of the application code
COPY . /app

# 7. Security
RUN useradd -ms /bin/bash appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "app.py"]
