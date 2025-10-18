# 🔍 Fake News Detection with RoBERTa + Docker

A containerized machine learning application for detecting fake news using transformer models (RoBERTa), built with Flask and Docker for easy deployment and team collaboration.

## 🎯 Quick Start

**For Complete Beginners (New to Docker):**
1. Read the [Docker Guide](docker-guide.md) first
2. Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
3. Clone this repository
4. Run the setup script

**For Experienced Users:**
```bash
git clone <your-repo>
cd fake-news-detection
./setup.sh start-dev
# App running at http://localhost:8000
```

## 📋 Table of Contents

1. [Features](#-features)
2. [Prerequisites](#-prerequisites)
3. [Installation & Setup](#-installation--setup)
4. [Usage](#-usage)
5. [API Documentation](#-api-documentation)
6. [Team Collaboration](#-team-collaboration)
7. [Deployment](#-deployment)
8. [Troubleshooting](#-troubleshooting)
9. [Contributing](#-contributing)

## ✨ Features

- **RoBERTa-based ML Model**: State-of-the-art transformer for text classification
- **RESTful API**: Easy-to-use endpoints for fake news detection
- **Docker Containerization**: Consistent environments across all machines
- **Multi-environment Support**: Development, testing, and production configurations
- **Batch Processing**: Analyze multiple texts simultaneously
- **Health Monitoring**: Built-in health checks and monitoring
- **Team-Ready**: Easy sharing and collaboration setup
- **Caching**: Redis integration for improved performance
- **Production-Ready**: Nginx reverse proxy and resource limits

## 🔧 Prerequisites

### Must Have:
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Git** for version control
- **4GB+ RAM** (for ML model loading)
- **Internet connection** (for downloading models)

### Nice to Have:
- **Python 3.9+** (for local development)
- **Visual Studio Code** with Docker extension
- **Postman** or **curl** for API testing

## 🚀 Installation & Setup

### Step 1: Install Docker

**Windows/Mac:**
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and restart your computer
3. Open Docker Desktop and complete setup

**Linux (Ubuntu/Debian):**
```bash
# Quick install script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

**Verify Installation:**
```bash
docker --version
docker compose version
```

### Step 2: Clone and Setup Project

```bash
# Clone the repository
git clone <your-project-repository>
cd fake-news-detection

# Make setup script executable (Linux/Mac)
chmod +x setup.sh

# Copy environment configuration
cp .env.example .env
# Edit .env if needed
```

### Step 3: Start the Application

```bash
# Option 1: Using setup script (Recommended)
./setup.sh start-dev

# Option 2: Using Docker Compose directly
docker-compose -f docker-compose.dev.yml up --build -d

# Option 3: Manual Docker commands
docker build -t fake-news-detector .
docker run -d -p 8000:8000 fake-news-detector
```

### Step 4: Verify Installation

```bash
# Check if containers are running
docker ps

# Test the API
curl http://localhost:8000/health

# View the web interface
# Open http://localhost:8000 in your browser
```

## 📖 Usage

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
      "SHOCKING: You won't believe what happened next!",
      "Government announces new policy changes."
    ]
  }'
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

### Python Client Example
```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', json={
    'text': 'Your news article text here'
})
result = response.json()
print(f"Prediction: {result['result']['prediction']}")
print(f"Confidence: {result['result']['confidence']:.2f}")

# Batch prediction
response = requests.post('http://localhost:8000/batch_predict', json={
    'texts': ['Text 1', 'Text 2', 'Text 3']
})
results = response.json()
for item in results['results']:
    print(f"Text {item['index']}: {item['result']['prediction']}")
```

## 🔌 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API documentation homepage |
| GET | `/health` | Health check and status |
| GET | `/stats` | System and model statistics |
| POST | `/predict` | Analyze single text |
| POST | `/batch_predict` | Analyze multiple texts |

### Request/Response Examples

**POST /predict**
```json
// Request
{
  "text": "Your news article text to analyze"
}

// Response
{
  "success": true,
  "result": {
    "prediction": "FAKE",
    "confidence": 0.85,
    "fake_probability": 0.85,
    "real_probability": 0.15,
    "text_length": 156,
    "model_used": "roberta-base"
  },
  "timestamp": "2025-01-15T10:30:00"
}
```

**POST /batch_predict**
```json
// Request
{
  "texts": ["Text 1", "Text 2", "Text 3"]
}

// Response
{
  "success": true,
  "results": [
    {
      "index": 0,
      "text_preview": "Text 1",
      "result": {
        "prediction": "REAL",
        "confidence": 0.92
      }
    }
  ],
  "total_processed": 3
}
```

## 👥 Team Collaboration

### Sharing with Team Members

**Method 1: Repository Sharing (Recommended)**
```bash
# Team member workflow:
git clone <your-repo>
cd fake-news-detection
./setup.sh start-dev
# Ready to work!
```

**Method 2: Docker Hub Sharing**
```bash
# Push to Docker Hub
docker build -t yourusername/fake-news-detector:v1.0 .
docker push yourusername/fake-news-detector:v1.0

# Team members pull and run
docker pull yourusername/fake-news-detector:v1.0
docker run -d -p 8000:8000 yourusername/fake-news-detector:v1.0
```

**Method 3: Image Export/Import**
```bash
# Export image to file
docker save fake-news-detector | gzip > fake-news-app.tar.gz
# Share the .tar.gz file

# Import on other machines
gunzip -c fake-news-app.tar.gz | docker load
docker run -d -p 8000:8000 fake-news-detector
```

### Development Workflow

```bash
# Daily development
./setup.sh start-dev    # Start development environment
# Make code changes...
./setup.sh logs         # View logs for debugging
./setup.sh test         # Test API endpoints
./setup.sh stop         # Stop when done

# Code updates
./setup.sh restart-dev  # Restart after changes
./setup.sh status       # Check container status
```

### Environment Management

- **Development**: `docker-compose.dev.yml` - Hot reload, debug logging
- **Production**: `docker-compose.prod.yml` - Optimized, reverse proxy
- **Testing**: Isolated containers for CI/CD

## 🚀 Deployment

### Development Environment
```bash
./setup.sh start-dev
# Features: Hot reload, debug logging, development tools
```

### Production Environment
```bash
./setup.sh start-prod
# Features: Nginx reverse proxy, resource limits, optimized settings
```

### Cloud Deployment

**AWS EC2/DigitalOcean/Any VPS:**
```bash
# On your server
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER

git clone <your-repo>
cd fake-news-detection
./setup.sh start-prod
```

**Docker Swarm (Multi-server):**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml fake-news-stack
```

**Kubernetes:**
```bash
# Convert compose to k8s (using kompose)
kompose convert -f docker-compose.prod.yml
kubectl apply -f fake-news-detector-deployment.yaml
```

## 🔧 Troubleshooting

### Common Issues & Solutions

**1. Port Already in Use**
```bash
# Find and stop process using port 8000
sudo lsof -i :8000
sudo kill -9 <PID>

# Or use different port
APP_PORT=8001 ./setup.sh start-dev
```

**2. Docker Permission Denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

**3. Out of Memory/Disk Space**
```bash
# Clean up Docker resources
docker system prune -a -f
./setup.sh cleanup
```

**4. Model Download Fails**
```bash
# Check internet connection
# Set proxy if needed
docker run -e HTTP_PROXY=http://proxy:8080 fake-news-detector

# Or pre-download models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')"
```

**5. Container Won't Start**
```bash
# Check logs
./setup.sh logs
docker logs fake_news_app

# Debug interactively
docker run -it fake-news-detector /bin/bash
```

### Debugging Commands
```bash
# Container status
docker ps -a

# Resource usage
docker stats

# Execute commands in container
docker exec -it fake_news_app /bin/bash

# View Docker system info
docker system info

# Network troubleshooting
docker network ls
docker network inspect fake-news-network
```

### Getting Help

1. **Check the logs**: `./setup.sh logs`
2. **Review Docker Guide**: [docker-guide.md](docker-guide.md)
3. **Test API endpoints**: `./setup.sh test`
4. **Check system resources**: `docker stats`
5. **Consult documentation**: [Docker Docs](https://docs.docker.com/)

## 📝 Configuration

### Environment Variables (.env)
```bash
# Application
APP_PORT=8000
FLASK_ENV=development
LOG_LEVEL=INFO

# Model
MODEL_NAME=roberta-base
CACHE_SIZE=100

# Redis (optional)
REDIS_HOST=redis
REDIS_PORT=6379

# Production
WORKERS=4
NGINX_PORT=80
```

### Customization
- **Models**: Change `MODEL_NAME` in `.env`
- **Ports**: Modify `APP_PORT` for different ports
- **Resources**: Adjust memory/CPU limits in compose files
- **Caching**: Enable/disable Redis caching
- **Logging**: Set log levels and output formats

## 🛠 Development

### Local Development Setup
```bash
# Install Python dependencies locally (optional)
pip install -r requirements.txt

# Run without Docker (for development)
python app.py

# With Docker (recommended)
./setup.sh start-dev
```

### Adding New Features
1. Modify code in your IDE
2. Test changes: `./setup.sh restart-dev`
3. Check logs: `./setup.sh logs`
4. Test API: `./setup.sh test`
5. Commit changes to Git

### Testing
```bash
# API testing
./setup.sh test

# Manual testing
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"test"}'

# Load testing (optional)
pip install locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📊 Monitoring & Maintenance

### Health Monitoring
```bash
# Check application health
curl http://localhost:8000/health

# View system stats
curl http://localhost:8000/stats

# Monitor resources
docker stats
```

### Log Management
```bash
# View recent logs
./setup.sh logs

# Follow logs in real-time
docker-compose logs -f

# Log rotation (production)
docker run --log-driver=json-file --log-opt max-size=10m --log-opt max-file=3 ...
```

### Backup & Recovery
```bash
# Backup models and data
docker run --rm -v fake_news_model_cache:/data -v $(pwd):/backup ubuntu tar czf /backup/models-backup.tar.gz /data

# Backup database (if using)
docker exec fake_news_db pg_dump -U user dbname > backup.sql
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test: `./setup.sh start-dev`
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a Pull Request

### Development Guidelines
- Use Docker for development
- Follow PEP 8 for Python code
- Add tests for new features
- Update documentation
- Test with different environments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the ML models
- [Docker](https://www.docker.com/) for containerization platform
- [Flask](https://flask.palletsprojects.com/) for the web framework
- RoBERTa model authors for the transformer architecture

---

## 🆘 Need Help?

**For Docker beginners**: Read [docker-guide.md](docker-guide.md)

**For quick help**:
```bash
./setup.sh help
```

**For issues**:
1. Check logs: `./setup.sh logs`
2. Test API: `./setup.sh test`
3. Review troubleshooting section above
4. Create GitHub issue with logs and system info

**Contact**: [Your contact information]

---

**Happy fake news detecting! 🔍🐳**