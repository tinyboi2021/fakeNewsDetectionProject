# 🚀 Complete Setup & Usage Guide
## Fake News & Clickbait Detection Project with Docker + GPU

**The Ultimate All-in-One Guide for Setup, Training, Deployment, and Team Collaboration**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites & Hardware Requirements](#prerequisites--hardware-requirements)
3. [Installation & Environment Setup](#installation--environment-setup)
4. [Docker Setup with GPU Support](#docker-setup-with-gpu-support)
5. [Production Training Pipeline](#production-training-pipeline)
6. [Running the Application](#running-the-application)
7. [Jupyter Notebook Access](#jupyter-notebook-access)
8. [Flask API Usage](#flask-api-usage)
9. [Git LFS for Large Files](#git-lfs-for-large-files)
10. [Team Collaboration Workflow](#team-collaboration-workflow)
11. [Troubleshooting & Common Issues](#troubleshooting--common-issues)
12. [Docker Commands Reference](#docker-commands-reference)
13. [Best Practices & Production Tips](#best-practices--production-tips)

---

## 🎯 Project Overview

A production-grade containerized machine learning system for detecting fake news and clickbait using transformer models (RoBERTa), built with:

- **GPU-accelerated training** with NVIDIA CUDA 12.1
- **K-fold cross-validation** for robust model evaluation
- **Docker containerization** for consistent deployment
- **Flask REST API** for easy integration
- **Jupyter integration** for development
- **Git LFS** for large model files
- **Interactive training** with real-time feedback

### Key Features

✅ **RoBERTa-based ML Model** - State-of-the-art transformer  
✅ **GPU Acceleration** - 10-20x faster training  
✅ **Production-Ready API** - RESTful endpoints with health checks  
✅ **K-fold Cross-Validation** - 5-fold CV for reliable metrics  
✅ **Hardware Profiles** - Optimized configs for RTX 4060 Ti and A100  
✅ **Interactive Mode** - Test predictions in real-time  
✅ **Batch Processing** - Analyze multiple texts simultaneously  
✅ **Session Logging** - Automatic prediction tracking  

---

## 🔧 Prerequisites & Hardware Requirements

### Must Have

**Software:**
- **Docker Desktop 20.10+** with GPU support (Windows/Mac) or Docker Engine (Linux)
- **NVIDIA GPU** with CUDA support (RTX 3000+, A100, etc.)
- **NVIDIA Container Toolkit** for Docker GPU access
- **Git + Git LFS** for version control and large files
- **Python 3.8+** (for local development)

**Hardware:**
- **Minimum:** 8GB RAM, 6GB GPU VRAM (RTX 4060 Ti), 20GB storage
- **Recommended:** 16GB+ RAM, 16GB+ GPU VRAM (A100), 50GB storage
- **CPU:** 4+ cores minimum, 8+ recommended

### Nice to Have

- Visual Studio Code with Docker extension
- Postman or curl for API testing
- Windows Terminal or iTerm2 for better CLI experience

### Verify Your System

```bash
# Check Docker
docker --version
docker compose version

# Check NVIDIA GPU
nvidia-smi

# Check Git LFS
git lfs version

# Check Python
python --version
```

---

## 📦 Installation & Environment Setup

### Step 1: Install Docker with GPU Support

**Windows:**

1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and restart computer
3. Open Docker Desktop → Settings → Resources → WSL Integration
4. Enable "Use WSL 2 based engine"
5. Enable GPU support in Docker settings

**Linux (Ubuntu/Debian):**

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -fsSL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Logout and login again
```

**Verify GPU Access:**

```bash
# Test NVIDIA Docker integration
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Should display GPU information
```

### Step 2: Install Git LFS

**Windows:**
```bash
choco install git-lfs    # With Chocolatey
# or download from https://git-lfs.github.com/
```

**Mac:**
```bash
brew install git-lfs
```

**Linux:**
```bash
sudo apt-get install git-lfs
```

**Initialize:**
```bash
git lfs install
```

### Step 3: Clone Repository

```bash
# Clone with LFS
git clone <your-repository-url>
cd fake-news-detection

# Pull LFS files
git lfs pull

# Make setup script executable (Linux/Mac)
chmod +x setup.sh

# Copy environment template
cp .env.example .env
```

---

## 🐳 Docker Setup with GPU Support

### Project Structure

```
fake-news-detection/
├── Dockerfile                          # GPU-enabled container definition
├── docker-compose.dev.yml             # Development configuration
├── requirements.txt                   # Python dependencies
├── production_fake_news_detector.py   # Main training script
├── app.py                            # Flask API
├── src/
│   ├── dataset1/News_dataset/        # Training datasets
│   ├── dataset2/News_dataset/        # Test datasets
│   └── Clickbait_dataset/            # Clickbait data
└── trainableModel/                   # Model storage
```

### Enhanced Dockerfile

Our Dockerfile uses NVIDIA CUDA 12.1 base with GPU support:

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
LABEL org.opencontainers.image.authors="Fake News Detection Team"

# Install system dependencies + nano, vim, Jupyter
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git curl nano vim build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Copy requirements first for Docker cache efficiency
COPY requirements.txt ./

# Install PyTorch with CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install Jupyter + other dependencies
RUN pip3 install --no-cache-dir \
    jupyter jupyterlab notebook ipykernel ipywidgets \
    -r requirements.txt

# Copy application code
COPY . /app

# Create non-root user
RUN id -u appuser || useradd -ms /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Set up Jupyter kernel
RUN mkdir -p /home/appuser/.local/share/jupyter/kernels/fake_news_env && \
    python3 -m ipykernel install \
      --prefix=/home/appuser/.local \
      --name fake_news_env \
      --display-name "Python (Fake News Detection)"

ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes
ENV PATH=/home/appuser/.local/bin:$PATH

EXPOSE 8000 8888

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "app.py"]
```

### docker-compose.dev.yml Configuration

```yaml
services:
  fake-news-detector:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: fake_news_app_dev
    ports:
      - "8000:8000"    # Flask API
      - "8888:8888"    # Jupyter
    environment:
      - FLASK_ENV=development
      - LOG_LEVEL=DEBUG
      - MODEL_NAME=roberta-base
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./app.py:/app/app.py
      - ./src:/app/src
      - ./trainableModel:/app/trainableModel
      - model_cache:/app/.cache/transformers
      - huggingface_cache:/app/.cache/huggingface
      - ./logs:/app/logs
      - pip_cache:/root/.cache/pip
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - fake-news-network
    restart: unless-stopped
    command: python3 app.py

  redis:
    image: redis:7-alpine
    container_name: fake_news_redis_dev
    ports:
      - "6379:6379"
    volumes:
      - redis_data_dev:/data
    networks:
      - fake-news-network
    restart: unless-stopped

volumes:
  model_cache:
  huggingface_cache:
  redis_data_dev:
  pip_cache:

networks:
  fake-news-network:
    driver: bridge
```

### Build and Start

```bash
# Build without cache (first time or after Dockerfile changes)
docker-compose -f docker-compose.dev.yml build --no-cache

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Verify containers are running
docker ps

# Check GPU access inside container
docker exec -it fake_news_app_dev python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

**Expected output:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

---

## 🎓 Production Training Pipeline

### Training Script Overview

The `production_fake_news_detector.py` script provides:

- **5-fold cross-validation** for robust evaluation
- **Class imbalance handling** with weighted loss
- **Early stopping** (patience=2) to prevent overfitting
- **Gradient clipping** for training stability
- **Learning rate scheduling** with warmup
- **Comprehensive metrics** (Accuracy, F1, ROC-AUC, Brier Score)
- **Automatic checkpointing** per fold
- **Interactive prediction mode** after training

### Hardware Profiles

**Low Profile (RTX 4060 Ti - 6GB VRAM):**
```bash
python production_fake_news_detector.py \
  --csv /app/src/dataset2/News_dataset/train_dataset.csv \
  --profile low
```
- batch_size=8
- max_length=128
- num_epochs=3
- Training time: ~2-3 hours

**High Profile (A100 - 40GB VRAM):**
```bash
python production_fake_news_detector.py \
  --csv /app/src/dataset2/News_dataset/train_dataset.csv \
  --profile high
```
- batch_size=32
- max_length=512
- num_epochs=5
- Training time: ~45-60 minutes

### Step-by-Step Training

**1. Access Container:**
```bash
docker exec -it fake_news_app_dev bash
cd /app/trainableModel
```

**2. Start Training:**
```bash
python production_fake_news_detector.py \
  --csv /app/src/dataset2/News_dataset/train_dataset.csv \
  --profile low
```

**3. Training Output:**
```
📂 Loading data...
✅ Loaded 44898 articles
   - Real news: 21417
   - Fake news: 23481

🚀 Device: cuda

============================================================
Training Fold 1/5
============================================================
Fold1 E1: TrL=0.3241 VaL=0.2145 VA=0.9234 VF1=0.9228 VAUC=0.9567
Fold1 E2: TrL=0.1892 VaL=0.1823 VA=0.9347 VF1=0.9342 VAUC=0.9723
Fold1 E3: TrL=0.1534 VaL=0.1756 VA=0.9389 VF1=0.9385 VAUC=0.9756

✅ Fold 1 Results:
   Accuracy: 0.9389
   F1-Score: 0.9385
   ROC-AUC: 0.9756
   Brier Score: 0.0498

[... Folds 2-5 ...]

============================================================
AVG METRICS: {'accuracy': 0.9367, 'f1_score': 0.9363, 'roc_auc': 0.9741, 'brier_score': 0.0512}
============================================================

💾 Best model saved to ./production_model/best_model
```

**4. Interactive Mode:**
```
Interactive? (y/n): y

Interactive mode: 'exit' to quit
Text: Breaking news: Scientists discover revolutionary technology!
{'text': '...', 'prediction': 'Fake', 'confidence': 0.9234, 'fake_probability': 0.9234, 'real_probability': 0.0766}

Text: exit
Logged 5 preds to ./results/session_20251018.txt
```

### Monitoring Training

```bash
# Monitor GPU usage
watch -n1 nvidia-smi

# View training logs
docker logs -f fake_news_app_dev

# Check disk usage
docker exec fake_news_app_dev df -h
```

---

## 🚀 Running the Application

### Start Flask API

The Flask API starts automatically with docker-compose, or manually:

```bash
# Inside container
python app.py

# Or from host
docker-compose -f docker-compose.dev.yml up
```

### Access Points

- **API Documentation**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Stats**: http://localhost:8000/stats
- **Jupyter Lab**: http://localhost:8888 (see next section)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API documentation |
| GET | `/health` | Health check |
| GET | `/stats` | System & GPU statistics |
| POST | `/predict` | Single text prediction |
| POST | `/batch_predict` | Batch prediction |
| POST | `/detect_clickbait` | Clickbait detection |
| POST | `/analyze_content` | Combined analysis |

### Testing API

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "SHOCKING: You wont believe what happened next!"}'
```

**Response:**
```json
{
  "success": true,
  "result": {
    "prediction": "FAKE",
    "confidence": 0.9234,
    "fake_probability": 0.9234,
    "real_probability": 0.0766,
    "device_used": "cuda:0"
  },
  "timestamp": "2025-10-18T08:30:00"
}
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

---

## 📓 Jupyter Notebook Access

### Start Jupyter Lab

```bash
# Inside container
docker exec -it fake_news_app_dev bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app
```

**Output:**
```
[I 2025-10-18 08:30:00.123 ServerApp] Jupyter Server 2.7.0 is running at:
[I 2025-10-18 08:30:00.123 ServerApp] http://fake_news_app_dev:8888/lab?token=abc123def456...
```

### Access from Windows Browser

1. Copy the token from terminal
2. Open http://localhost:8888
3. Paste token when prompted
4. Start editing Python files live!

### Using Jupyter for Development

- Navigate to `/app/trainableModel` folder
- Open `production_fake_news_detector.py`
- Edit code directly in browser
- Changes are synced via Docker volumes
- Run cells interactively for debugging

### Jupyter Features

✅ Full Python environment with all dependencies  
✅ GPU access for testing models  
✅ Live code editing with auto-save  
✅ Terminal access within browser  
✅ File browser and upload/download  
✅ Markdown cells for documentation  

---

## 🌐 Flask API Usage

### Comprehensive API Examples

**1. Health Check:**
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "timestamp": "2025-10-18T08:30:00"
}
```

**2. System Stats:**
```bash
curl http://localhost:8000/stats
```
```json
{
  "model_info": {
    "name": "roberta-base",
    "device": "cuda:0",
    "gpu_memory_used": "2.1GB"
  },
  "system_info": {
    "gpu_name": "NVIDIA GeForce RTX 4060 Ti",
    "cuda_version": "12.1"
  }
}
```

**3. Python Client:**
```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', 
                        json={'text': 'Your news article'})
result = response.json()
print(f"Prediction: {result['result']['prediction']}")
print(f"Confidence: {result['result']['confidence']:.2%}")

# Batch prediction
texts = ["Article 1", "Article 2", "Article 3"]
response = requests.post('http://localhost:8000/batch_predict',
                        json={'texts': texts})
for item in response.json()['results']:
    print(f"Text {item['index']}: {item['result']['prediction']}")
```

---

## 📦 Git LFS for Large Files

### Why Git LFS?

Model files (`.bin`, `.pt`) can be 500MB-2GB. Git LFS tracks these efficiently without bloating your repository.

### Setup Git LFS

```bash
# Install and initialize
git lfs install

# Track large file patterns (already in .gitattributes)
git lfs track "src/dataset1/News_dataset/**"
git lfs track "src/dataset2/News_dataset/**"
git lfs track "src/Clickbait_dataset/**"
git lfs track "trainableModel/**/*.bin"
git lfs track "trainableModel/**/*.pt"
git lfs track "production_model/**"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for large files"
```

### Working with LFS Files

**Push Large Files:**
```bash
# Add trained models
git add production_model/
git add trainableModel/
git commit -m "Add trained models via LFS"
git push origin main

# LFS uploads large files to LFS storage automatically
```

**Clone with LFS:**
```bash
# Clone repository
git clone <repo-url>
cd fake-news-detection

# Pull LFS files
git lfs pull

# Verify LFS files
git lfs ls-files
```

**Check LFS Status:**
```bash
# List tracked files
git lfs ls-files

# Check storage usage
git lfs fsck

# Prune old versions
git lfs prune
```

### LFS Best Practices

✅ Track all model files (.bin, .pt, .h5)  
✅ Track large datasets (.csv > 100MB)  
✅ Always `git lfs pull` after cloning  
✅ Use `git lfs migrate` for existing repos  
✅ Check LFS quota on GitHub/GitLab  

---

## 👥 Team Collaboration Workflow

### Onboarding New Team Members

**Step 1: Clone Repository**
```bash
git clone <your-repo-url>
cd fake-news-detection
git lfs pull
```

**Step 2: Build Environment**
```bash
# Build and start
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up -d

# Verify GPU
docker exec fake_news_app_dev python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Step 3: Ready to Develop!**
```bash
# Access container
docker exec -it fake_news_app_dev bash

# Start Jupyter
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Or train model
cd /app/trainableModel
python production_fake_news_detector.py --csv /app/src/dataset2/News_dataset/train_dataset.csv --profile low
```

### Sharing Trained Models

**Method 1: Git LFS (Recommended)**
```bash
# Team member trains model
python production_fake_news_detector.py --csv /app/src/dataset2/News_dataset/train_dataset.csv --profile low

# Commit and push (LFS handles large files)
git add production_model/
git commit -m "Add trained model - Accuracy: 93.67%"
git push

# Other team members pull
git pull
git lfs pull
```

**Method 2: Docker Hub**
```bash
# Build and push image with trained model
docker build -t yourusername/fake-news-detector:v1.0 .
docker push yourusername/fake-news-detector:v1.0

# Team members pull
docker pull yourusername/fake-news-detector:v1.0
```

**Method 3: Export/Import**
```bash
# Export trained model
docker run --rm -v $(pwd)/production_model:/models \
  ubuntu tar czf /models/trained_model_v1.tar.gz /models/best_model

# Share trained_model_v1.tar.gz file
# Team members extract
tar xzf trained_model_v1.tar.gz -C production_model/
```

### Development Workflow

**Daily Development:**
```bash
# Start environment
docker-compose up -d

# Make code changes
# (Files are synced via volumes)

# Test changes
docker exec fake_news_app_dev python app.py

# View logs
docker-compose logs -f

# Stop when done
docker-compose down
```

**Code Review:**
```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes and test
docker-compose -f docker-compose.dev.yml up --build

# Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-model

# Create Pull Request
```

---

## 🔧 Troubleshooting & Common Issues

### GPU Not Detected

**Issue:** Container shows `CUDA: False`

**Solution:**
```bash
# 1. Verify host GPU
nvidia-smi

# 2. Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# 3. Check docker-compose GPU config
# Ensure deploy.resources.reservations.devices is set correctly

# 4. Restart Docker Desktop (Windows)
# Settings → Resources → Enable GPU

# 5. Check NVIDIA Container Toolkit (Linux)
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### CUDA Out of Memory

**Issue:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size and sequence length
python production_fake_news_detector.py \
  --csv /app/src/dataset2/News_dataset/train_dataset.csv \
  --profile low

# Or manually edit CONFIG in script:
# CONFIG['batch_size'] = 4  # Instead of 8
# CONFIG['max_length'] = 64  # Instead of 128

# Monitor GPU memory
watch -n1 nvidia-smi
```

### Dataset Not Found

**Issue:** `KeyError: 'content'` or file not found

**Solution:**
```bash
# Check dataset structure
docker exec fake_news_app_dev ls -la /app/src/dataset2/News_dataset/

# Verify CSV columns
docker exec fake_news_app_dev head /app/src/dataset2/News_dataset/train_dataset.csv

# Ensure CSV has 'content' and 'is_fakenews' columns
# Update --csv path if needed
```

### Docker Build Fails

**Issue:** Build errors or cache issues

**Solution:**
```bash
# Clean build with no cache
docker-compose -f docker-compose.dev.yml build --no-cache

# Clean Docker system
docker system prune -a -f

# Remove volumes
docker-compose -f docker-compose.dev.yml down -v

# Rebuild
docker-compose -f docker-compose.dev.yml up --build
```

### Port Already in Use

**Issue:** Port 8000 or 8888 already in use

**Solution:**
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Or change port in docker-compose.dev.yml
ports:
  - "8001:8000"  # Change host port
  - "8889:8888"
```

### LFS Files Not Downloaded

**Issue:** Large files show as pointers

**Solution:**
```bash
# Pull LFS files
git lfs pull

# Force fetch all LFS objects
git lfs fetch --all

# Check LFS status
git lfs ls-files

# If still issues, reinstall LFS
git lfs install --force
```

### Permission Denied

**Issue:** Cannot write to mounted volumes

**Solution:**
```bash
# Check ownership
docker exec fake_news_app_dev ls -la /app

# Fix permissions (Linux/Mac)
docker exec -u 0 fake_news_app_dev chown -R appuser:appuser /app

# Or rebuild with correct user
docker-compose -f docker-compose.dev.yml build --no-cache
```

---

## 📘 Docker Commands Reference

### Container Management

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Start container
docker start fake_news_app_dev

# Stop container
docker stop fake_news_app_dev

# Restart container
docker restart fake_news_app_dev

# Remove container
docker rm fake_news_app_dev

# Execute command in container
docker exec fake_news_app_dev <command>

# Interactive shell
docker exec -it fake_news_app_dev bash

# View logs
docker logs fake_news_app_dev
docker logs -f fake_news_app_dev  # Follow

# Copy files
docker cp file.txt fake_news_app_dev:/app/
docker cp fake_news_app_dev:/app/results/ ./local_results/

# View resource usage
docker stats
```

### Image Management

```bash
# List images
docker images

# Build image
docker build -t fake-news-detector .

# Remove image
docker rmi fake-news-detector

# Prune unused images
docker image prune -a

# Tag image
docker tag fake-news-detector:latest yourusername/fake-news:v1.0

# Push to registry
docker push yourusername/fake-news:v1.0

# Pull from registry
docker pull yourusername/fake-news:v1.0
```

### Docker Compose Commands

```bash
# Start services
docker-compose -f docker-compose.dev.yml up

# Start in background
docker-compose -f docker-compose.dev.yml up -d

# Stop services
docker-compose -f docker-compose.dev.yml down

# Stop and remove volumes
docker-compose -f docker-compose.dev.yml down -v

# Rebuild and start
docker-compose -f docker-compose.dev.yml up --build

# View logs
docker-compose -f docker-compose.dev.yml logs
docker-compose -f docker-compose.dev.yml logs -f service_name

# List services
docker-compose -f docker-compose.dev.yml ps

# Execute in service
docker-compose -f docker-compose.dev.yml exec fake-news-detector bash

# Scale services
docker-compose -f docker-compose.dev.yml up --scale fake-news-detector=3
```

### System Maintenance

```bash
# Clean up everything
docker system prune -a -f

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# View disk usage
docker system df

# Inspect container
docker inspect fake_news_app_dev

# View container processes
docker top fake_news_app_dev
```

---

## 🎯 Best Practices & Production Tips

### Development Best Practices

✅ **Always use volumes** for code during development  
✅ **Use .dockerignore** to exclude unnecessary files  
✅ **Leverage build cache** with proper layer ordering  
✅ **Use multi-stage builds** for smaller images  
✅ **Set resource limits** to prevent system overload  
✅ **Enable health checks** for container monitoring  
✅ **Use non-root user** for security  

### Training Best Practices

✅ **Start with default config** before optimization  
✅ **Monitor GPU memory** with `nvidia-smi`  
✅ **Save checkpoints regularly** during training  
✅ **Use K-fold validation** for robust metrics  
✅ **Track experiments** with logs and metrics  
✅ **Test on validation set** before production  

### Production Deployment

✅ **Use production docker-compose** with optimizations  
✅ **Enable Nginx reverse proxy** for load balancing  
✅ **Set up monitoring** (Prometheus, Grafana)  
✅ **Implement rate limiting** on API endpoints  
✅ **Use environment variables** for secrets  
✅ **Enable HTTPS** with Let's Encrypt  
✅ **Set up automatic backups** for models and data  
✅ **Configure log rotation** to manage disk space  

### Security Best Practices

✅ **Never run as root** in containers  
✅ **Use specific image tags** (not `latest`)  
✅ **Scan images** for vulnerabilities  
✅ **Keep dependencies updated** regularly  
✅ **Use secrets management** (Docker secrets, Vault)  
✅ **Enable Docker Content Trust** (DCT)  
✅ **Restrict network access** with firewalls  

### Team Collaboration Tips

✅ **Document everything** in README files  
✅ **Use Git LFS** for large files  
✅ **Create feature branches** for development  
✅ **Review code** before merging  
✅ **Maintain consistent environments** with Docker  
✅ **Share trained models** via LFS or registry  
✅ **Keep .env.example** updated with required vars  

---

## 🚀 Quick Command Reference

### Most Used Commands

```bash
# Build and start everything
docker-compose -f docker-compose.dev.yml up --build -d

# Access container
docker exec -it fake_news_app_dev bash

# Train model (inside container)
cd /app/trainableModel
python production_fake_news_detector.py --csv /app/src/dataset2/News_dataset/train_dataset.csv --profile low

# Start Jupyter
docker exec -it fake_news_app_dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop everything
docker-compose -f docker-compose.dev.yml down

# Test API
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"test"}'

# Monitor GPU
watch -n1 nvidia-smi

# Git LFS pull
git lfs pull

# Clean Docker
docker system prune -a -f
```

---

## 📊 Expected Performance Metrics

| Metric | Low Profile (RTX 4060 Ti) | High Profile (A100) |
|--------|---------------------------|---------------------|
| **Training Time** | 2-3 hours | 45-60 minutes |
| **Accuracy** | 93-95% | 94-96% |
| **F1-Score** | 0.93-0.95 | 0.94-0.96 |
| **ROC-AUC** | 0.94-0.97 | 0.95-0.98 |
| **Inference Time** | ~0.1s per text | ~0.05s per text |
| **Batch Size** | 8 | 32 |
| **GPU Memory** | ~4-5GB | ~12-16GB |

---

## 🆘 Getting Help

### Support Resources

- **GitHub Issues**: Report bugs with logs and system info
- **Docker Docs**: https://docs.docker.com/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Transformers Docs**: https://huggingface.co/docs/transformers/

### Diagnostic Commands

```bash
# System info
docker info
nvidia-smi

# Container diagnostics
docker exec fake_news_app_dev python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Check dependencies
docker exec fake_news_app_dev pip list | grep torch
docker exec fake_news_app_dev pip list | grep transformers

# Network diagnostics
docker network inspect fake-news-network

# Volume diagnostics
docker volume ls
docker volume inspect model_cache
```

---

## 🎉 Summary

You now have a **complete, production-ready fake news detection system** with:

✅ GPU-accelerated training (10-20x faster)  
✅ K-fold cross-validation for robust models  
✅ Docker containerization for consistency  
✅ Jupyter integration for development  
✅ Flask REST API for easy integration  
✅ Git LFS for efficient model versioning  
✅ Team collaboration workflows  
✅ Comprehensive troubleshooting guide  

### Next Steps

1. **Set up your environment** following Installation section
2. **Train your first model** with production pipeline
3. **Test the API** with sample requests
4. **Share with your team** using Git LFS
5. **Deploy to production** with docker-compose.prod.yml

**Happy detecting! 🎯🚀**

---

**Last Updated:** October 18, 2025  
**Version:** 2.0  
**Authors:** Fake News Detection Team