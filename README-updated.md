# 🔍 Fake News Detection with RoBERTa + Docker (GPU-Enabled)

A containerized machine learning application for detecting fake news using transformer models (RoBERTa), built with Flask and Docker with GPU acceleration for enhanced performance, designed for easy deployment and team collaboration.

## 🎯 Quick Start

**For Complete Beginners (New to Docker):**

1. Install Docker Desktop and NVIDIA Container Toolkit
2. Clone this repository
3. Run the setup script with GPU support

**For Experienced Users:**

```bash
git clone <your-repo>
cd fake-news-detection
./setup.sh start-dev
# App running at http://localhost:8000 with GPU acceleration
```

## 📋 Table of Contents

1. [Features](#-features)
2. [Prerequisites](#-prerequisites)
3. [Installation & Setup](#-installation--setup)
4. [GPU Setup](#-gpu-setup)
5. [Model Training](#-model-training)
6. [Usage](#-usage)
7. [API Documentation](#-api-documentation)
8. [Git LFS Setup](#-git-lfs-setup)
9. [Team Collaboration](#-team-collaboration)
10. [Deployment](#-deployment)
11. [Troubleshooting](#-troubleshooting)

## ✨ Features

- **RoBERTa-based ML Model**: State-of-the-art transformer for text classification
- **GPU Acceleration**: CUDA support for faster training and inference
- **Trainable Model**: Custom model training with your own datasets
- **RESTful API**: Easy-to-use endpoints for fake news detection
- **Docker Containerization**: Consistent environments across all machines
- **Interactive Training**: Pause/resume training with checkpoints
- **Real-time Testing**: Interactive input field for testing predictions
- **Result Logging**: Automatic saving of predictions to text files
- **Git LFS Integration**: Efficient handling of large model files
- **Multi-environment Support**: Development, testing, and production configurations
- **Batch Processing**: Analyze multiple texts simultaneously
- **Health Monitoring**: Built-in health checks and monitoring

## 🔧 Prerequisites

### Must Have:

- **Docker Desktop** (Windows/Mac) with GPU support enabled
- **NVIDIA GPU** with CUDA support (for training acceleration)
- **NVIDIA Container Toolkit** for Docker GPU access
- **Git LFS** for large file management
- **8GB+ RAM** (for ML model training)
- **Internet connection** (for downloading models and datasets)

### Nice to Have:

- **Python 3.9+** (for local development)
- **Visual Studio Code** with Docker extension
- **Postman** or **curl** for API testing

## 🚀 Installation & Setup

### Step 1: Install Docker with GPU Support

**Windows:**

1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and restart your computer
3. Open Docker Desktop → Settings → Resources → Enable "Use WSL2 based engine"
4. Enable GPU support in Docker Desktop settings

**Linux (Ubuntu/Debian):**

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Logout and login again
```

**Verify GPU Access:**

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Step 2: Install Git LFS

**Windows/Mac:**

```bash
# Download from https://git-lfs.github.io/ or use package managers
# For Windows with Chocolatey:
choco install git-lfs

# For Mac with Homebrew:
brew install git-lfs
```

**Linux:**

```bash
sudo apt-get install git-lfs
```

**Initialize Git LFS:**

```bash
git lfs install
```

### Step 3: Clone and Setup Project

```bash
# Clone the repository
git clone <your-project-repository>
cd fake-news-detection

# Pull LFS files
git lfs pull

# Make setup script executable (Linux/Mac)
chmod +x setup.sh

# Copy environment configuration
cp .env.example .env
```

## 🎮 GPU Setup

### Docker Configuration for GPU

Your `docker-compose.dev.yml` includes GPU support:

```yaml
services:
  fake-news-detector:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    device_requests:
      - driver: nvidia
        count: all
        capabilities: [gpu]
```

### Dockerfile with CUDA Support

The project uses NVIDIA CUDA base image:

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install PyTorch with CUDA explicitly
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install other requirements (without torch to avoid conflicts)
RUN pip3 install --no-cache-dir -r requirements.txt
```

### GPU Verification in Code

Your training script automatically detects and uses GPU:

```python
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU.")
```

## 🎓 Model Training

### Start Training Environment

```bash
# Start the development environment
./setup.sh start-dev

# Access the training container
docker exec -it fake_news_app_dev bash

# Start containers without rebuilding
docker-compose -f docker-compose.dev.yml up -d


# Navigate to training directory
cd /app/trainableModel

# Run the training script
python fake_news_roberta_detector.py
```

### Training Features

**Automatic Dataset Loading:**

- Uses datasets from `/app/src/dataset1/News_dataset/`
- Automatically combines Fake.csv and True.csv
- Preprocesses text data for optimal training

**Checkpoint Support:**

- Training can be paused and resumed
- Automatic checkpoint saving every epoch
- Resume from last checkpoint automatically

**Interactive Testing:**

- After training, enter news text to test predictions
- Real-time feedback on fake/real classification
- Confidence scores for each prediction

**Result Logging:**

- All predictions saved to timestamped text files
- Automatic logging for audit trails
- Easy result sharing and analysis

### Training Commands

```bash
# Basic training (inside container)
python fake_news_roberta_detector.py

# With custom parameters
python fake_news_roberta_detector.py --epochs 5 --batch-size 32

# Resume from checkpoint
python fake_news_roberta_detector.py --resume

# Monitor GPU usage during training
watch -n1 nvidia-smi
```

## 📖 Usage

### Starting the Application

```bash
# Option 1: Using setup script (Recommended)
./setup.sh start-dev

# Option 2: Using Docker Compose directly
docker-compose -f docker-compose.dev.yml up --build

# Verify GPU is available in container
docker exec -it fake_news_app_dev python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Web Interface

Open http://localhost:8000 in your browser to see the API documentation and test the endpoints.

### Command Line Testing

**Single Text Analysis:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking: Scientists discover revolutionary new technology that will change everything!"
  }'
```

**Batch Analysis:**

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is a legitimate news article about science.",
      "SHOCKING: You won'\''t believe what happened next!",
      "Government announces new policy changes."
    ]
  }'
```

**Health Check with GPU Status:**

```bash
curl http://localhost:8000/stats
```

### Training Your Own Model

```bash
# Access container
docker exec -it fake_news_app_dev bash

# Navigate to training folder
cd /app/trainableModel

# Start interactive training
python fake_news_roberta_detector.py

# The script will:
# 1. Load your dataset
# 2. Train the model using GPU
# 3. Save checkpoints
# 4. Allow interactive testing
# 5. Save predictions to files
```

## 🔌 API Documentation

### Endpoints

| Method | Endpoint         | Description                |
| ------ | ---------------- | -------------------------- |
| GET    | `/`              | API documentation homepage |
| GET    | `/health`        | Health check and status    |
| GET    | `/stats`         | System and GPU statistics  |
| POST   | `/predict`       | Analyze single text        |
| POST   | `/batch_predict` | Analyze multiple texts     |

### Enhanced Response Format

**POST /predict**

```json
{
  "success": true,
  "result": {
    "prediction": "FAKE",
    "confidence": 0.85,
    "fake_probability": 0.85,
    "real_probability": 0.15,
    "text_length": 156,
    "model_used": "roberta-base",
    "processing_time": 0.023,
    "device_used": "cuda:0"
  },
  "timestamp": "2025-10-13T17:30:00"
}
```

**GET /stats**

```json
{
  "model_info": {
    "name": "roberta-base",
    "loaded": true,
    "device": "cuda",
    "gpu_memory_used": "2.1GB"
  },
  "system_info": {
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 4060 Ti",
    "cuda_version": "12.1"
  }
}
```

## 📦 Git LFS Setup

### Tracking Large Files

The project automatically tracks large model files:

```bash
# Large files are already tracked in .gitattributes
fake_news_roberta/** filter=lfs diff=lfs merge=lfs -text
trainableModel/fake_news_roberta/** filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
```

### Working with LFS Files

```bash
# Clone with LFS files
git clone <repo-url>
git lfs pull

# Add new large files
git lfs track "path/to/large/file"
git add .gitattributes
git add path/to/large/file
git commit -m "Add large file with LFS"

# Push with LFS
git push origin main

# For existing large files in history
git lfs migrate import --include="*.bin,*.pt"
git push --force
```

### LFS Status and Management

```bash
# Check LFS status
git lfs ls-files

# Check LFS storage usage
git lfs fsck

# Prune old LFS objects
git lfs prune
```

## 👥 Team Collaboration

### Complete Team Setup

**Team Member Workflow:**

```bash
# 1. Clone repository
git clone <your-repo>
cd fake-news-detection

# 2. Pull LFS files
git lfs pull

# 3. Start development environment
./setup.sh start-dev

# 4. Access training environment
docker exec -it fake_news_app_dev bash
cd /app/trainableModel

# 5. Ready to train and develop!
```

### File Synchronization

The project includes live file synchronization:

```yaml
volumes:
  - ./app.py:/app/app.py
  - ./src:/app/src
  - ./trainableModel:/app/trainableModel # Live sync for training files
  - model_cache:/app/.cache/transformers
  - ./logs:/app/logs
  - ./data:/app/data
```

### Sharing Trained Models

```bash
# After training, models are automatically tracked with LFS
git add trainableModel/fake_news_roberta/
git commit -m "Add trained model"
git push

# Team members can pull the trained model
git lfs pull
```

## 🚀 Deployment

### Development with GPU

```bash
./setup.sh start-dev
# Features: GPU acceleration, hot reload, debug logging, live file sync
```

### Production with GPU

```bash
./setup.sh start-prod
# Features: GPU optimization, Nginx reverse proxy, resource limits
```

### Cloud Deployment with GPU

**AWS EC2 with GPU:**

```bash
# Launch GPU instance (p3.2xlarge, g4dn.xlarge, etc.)
# Install NVIDIA drivers and Docker

sudo apt update
sudo apt install -y nvidia-driver-470
sudo reboot

# Install Docker and NVIDIA Container Toolkit
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone and deploy
git clone <your-repo>
cd fake-news-detection
git lfs pull
./setup.sh start-prod
```

## 🔧 Troubleshooting

### GPU-Related Issues

**1. GPU Not Detected in Container**

```bash
# Check GPU support on host
nvidia-smi

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check container GPU access
docker exec -it fake_news_app_dev nvidia-smi
```

**2. CUDA Out of Memory**

```bash
# Reduce batch size in training
# Edit fake_news_roberta_detector.py:
# batch_size=8  # Instead of 16 or 32

# Monitor GPU memory
watch -n1 nvidia-smi
```

**3. LFS Files Not Downloaded**

```bash
# Pull LFS files
git lfs pull

# Check LFS status
git lfs ls-files

# Reset LFS if needed
git lfs fetch --all
```

**4. Docker Build Cache Issues**

```bash
# Clean build with no cache
docker-compose -f docker-compose.dev.yml build --no-cache

# Clean up Docker system
docker system prune -f
```

### Training Issues

**1. Model Files Corrupted**

```bash
# Remove broken model directory
rm -rf trainableModel/fake_news_roberta

# Restart training
docker exec -it fake_news_app_dev bash
cd /app/trainableModel
python fake_news_roberta_detector.py
```

**2. Dataset Not Found**

```bash
# Check dataset paths
ls -la /app/src/dataset1/News_dataset/

# Verify CSV files
head /app/src/dataset1/News_dataset/Fake.csv
head /app/src/dataset1/News_dataset/True.csv
```

**3. Permission Issues**

```bash
# Fix permissions
docker exec -u 0 -it fake_news_app_dev chown -R appuser:appuser /app
```

### Common Commands for Debugging

```bash
# Check container status and resources
docker stats

# Access container with root privileges
docker exec -u 0 -it fake_news_app_dev bash

# Check GPU memory usage
docker exec -it fake_news_app_dev nvidia-smi

# View training logs
docker logs fake_news_app_dev -f

# Check disk space
docker exec -it fake_news_app_dev df -h

# Monitor GPU during training
watch -n1 "docker exec fake_news_app_dev nvidia-smi"
```

## 📊 Performance Optimization

### GPU Memory Management

```python
# In your training code, add these optimizations:
torch.cuda.empty_cache()  # Clear GPU cache
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
```

### Docker Resource Limits

```yaml
# In docker-compose.yml, add resource limits:
deploy:
  resources:
    limits:
      memory: 8G
    reservations:
      memory: 4G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Training Optimization Tips

- Start with smaller batch sizes (8-16)
- Use gradient accumulation for larger effective batch sizes
- Enable mixed precision training for faster training
- Monitor GPU utilization with `nvidia-smi`
- Use checkpointing to resume interrupted training

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Clone and set up LFS: `git clone <fork> && git lfs pull`
3. Create feature branch: `git checkout -b feature-name`
4. Start development environment: `./setup.sh start-dev`
5. Make changes and test with GPU
6. Commit with LFS: `git add . && git commit -m "Add feature"`
7. Push: `git push origin feature-name`
8. Submit Pull Request

### Guidelines

- Test with both CPU and GPU configurations
- Ensure LFS files are properly tracked
- Add GPU memory usage information in PR description
- Test training functionality with sample data
- Update documentation for any new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the ML models
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) for GPU support
- [Docker](https://www.docker.com/) for containerization platform
- [Git LFS](https://git-lfs.github.io/) for large file management
- [Flask](https://flask.palletsprojects.com/) for the web framework

---

## 🆘 Need Help?

**For GPU setup issues**:

1. Verify NVIDIA drivers: `nvidia-smi`
2. Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
3. Ensure container has GPU access: `docker exec -it <container> nvidia-smi`

**For LFS issues**:

1. Check LFS installation: `git lfs version`
2. Pull LFS files: `git lfs pull`
3. Verify LFS tracking: `git lfs ls-files`

**For training issues**:

1. Check logs: `./setup.sh logs`
2. Monitor GPU: `watch -n1 nvidia-smi`
3. Access container: `docker exec -it fake_news_app_dev bash`

**Quick help**:

```bash
./setup.sh help
```

---

**Happy GPU-accelerated fake news detecting! 🔍🚀**
