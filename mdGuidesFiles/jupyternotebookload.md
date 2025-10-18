<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Fake News Detection Team"

# 1. Install system dependencies

RUN apt-get update \&\& apt-get install -y --no-install-recommends \
    python3-pip python3-setuptools python3-dev git curl \&\& \
    rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 3. COPY ONLY requirements.txt first for cache efficiency

COPY requirements.txt ./

# 4. Install PyTorch with CUDA from the official wheel

RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 5. Install all the other dependencies (they *should NOT* include torch)

RUN pip3 install --no-cache-dir -r requirements.txt

# 6. COPY the rest of the application code

COPY . /app

# 7. Security

RUN useradd -ms /bin/bash appuser \&\& chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "app.py"]

edit this file to also include line to install nano and jupyter notebook editor

Here's the enhanced Dockerfile with nano and Jupyter Notebook/JupyterLab installed:

## 🔧 **Key Changes Made:**

### **1. Added nano and vim**

```dockerfile
nano \
vim \
```

Now you can edit files directly inside the container using `nano` or `vim`.

### **2. Added Jupyter Ecosystem**

```dockerfile
jupyter \
jupyterlab \
notebook \
ipykernel \
ipywidgets
```

Complete Jupyter installation including:

- **Jupyter Notebook** - Classic notebook interface
- **JupyterLab** - Modern IDE-like interface
- **ipykernel** - Python kernel for notebooks
- **ipywidgets** - Interactive widgets


### **3. Set Up Jupyter Kernel**

```dockerfile
python3 -m ipykernel install --user --name fake_news_env --display-name "Python (Fake News Detection)"
```

Creates a custom kernel specifically for your project.

### **4. Added Jupyter Port**

```dockerfile
EXPOSE 8888
```

Default Jupyter port exposed alongside Flask (8000).

### **5. Set Jupyter Environment Variable**

```dockerfile
ENV JUPYTER_ENABLE_LAB=yes
```

Enables JupyterLab by default.

***

## 🚀 **How to Use:**

### **1. Rebuild Your Docker Image**

```bash
docker-compose -f docker-compose.dev.yml build --no-cache
```


### **2. Start the Container**

```bash
docker-compose -f docker-compose.dev.yml up
```


### **3. Use nano Inside Container**

```bash
docker exec -it fake_news_app_dev bash
nano /app/trainableModel/production_fake_news_detector.py
```


### **4. Start Jupyter Notebook**

```bash
# Inside container
docker exec -it fake_news_app_dev bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Or with JupyterLab:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```


### **5. Access Jupyter in Browser**

```
http://localhost:8888
```

Copy the token from terminal output.

***

## 📝 **Optional: docker-compose.yml Update**

To automatically start Jupyter, update your `docker-compose.dev.yml`:

```yaml
services:
  fake-news-detector:
    # ... existing config ...
    ports:
      - "8000:8000"
      - "8888:8888"  # Add Jupyter port
    
    # Optional: Override command to start both Flask and Jupyter
    # command: >
    #   bash -c "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
    #            python3 app.py"
```


***

## 🎯 **Quick Commands:**

| Task | Command |
| :-- | :-- |
| **Edit with nano** | `nano filename.py` |
| **Edit with vim** | `vim filename.py` |
| **Start Jupyter Notebook** | `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser` |
| **Start JupyterLab** | `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser` |
| **List kernels** | `jupyter kernelspec list` |
| **Access Jupyter** | Open `http://localhost:8888` in browser |

Now you have a fully-featured development environment with nano, vim, and Jupyter! 🎉

