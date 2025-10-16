FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL org.opencontainers.image.authors="Fake News Detection Team"

# 1. Install system dependencies including nano and Jupyter prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-dev \
    git \
    curl \
    nano \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip once
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 3. Copy only requirements first for Docker cache efficiency
COPY requirements.txt ./

# 4. Install PyTorch with CUDA from the official wheel
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 5. Install Jupyter Notebook, JupyterLab, and other dependencies
RUN pip3 install --no-cache-dir \
    jupyter jupyterlab notebook ipykernel ipywidgets \
    -r requirements.txt

# 6. Copy the rest of the application code
COPY . /app

# 7. Create non-root user and set permissions
RUN id -u appuser || useradd -ms /bin/bash appuser && \
    chown -R appuser:appuser /app

# 8. Switch to non-root user before Jupyter kernel install
USER appuser

# 9. Set up Jupyter kernel under appuser home
RUN mkdir -p /home/appuser/.local/share/jupyter/kernels/fake_news_env && \
    python3 -m ipykernel install \
    --prefix=/home/appuser/.local \
    --name fake_news_env \
    --display-name "Python (Fake News Detection)"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose application and Jupyter ports
EXPOSE 8000
EXPOSE 8888

# Healthcheck for Flask app
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python3", "app.py"]
