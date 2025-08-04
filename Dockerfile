# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
LABEL maintainer="Fake News Detection Team"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-setuptools python3-dev git curl && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
WORKDIR /app

COPY requirements.txt ./

# Install PyTorch with CUDA explicitly first
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Then install the rest of the requirements
RUN pip3 install --no-cache-dir -r requirements.txt


COPY . /app

RUN useradd -ms /bin/bash appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "app.py"]
