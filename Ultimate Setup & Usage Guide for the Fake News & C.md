<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Ultimate Setup \& Usage Guide for the Fake News \& Clickbait Detection Project

This comprehensive guide brings together all steps—environment setup, Docker, GPU profiles, training, inference, API usage, Jupyter integration, Git LFS, and troubleshooting—into a single reference.

## Table of Contents

1. Prerequisites
2. Repository Structure
3. Environment Setup
4. Docker \& GPU Profiles
5. Installing Dependencies
6. Git LFS for Large Files
7. Training \& Validation
8. Interactive Inference
9. Flask API Usage
10. Jupyter Notebook Access
11. Clickbait Detection Integration
12. Troubleshooting
13. Quick Command Reference

***

## 1. Prerequisites

- Hardware
- GPU: NVIDIA RTX 4060 Ti (6 GB VRAM) or A100 (40 GB)
- RAM: ≥ 8 GB (16 GB recommended)
- Disk: ≥ 20 GB free
- Software
- Docker ≥ 20.10 + NVIDIA Container Toolkit
- Python 3.8+ (for local installs)
- Git + Git LFS

***

## 2. Repository Structure

```
.
├── Dockerfile
├── docker-compose.dev.yml
├── requirements.txt
├── production_fake_news_detector.py
├── train_clickbait_detector.py
├── CLICKBAIT-DETECTION.md
├── PRODUCTION-DETECTOR-GUIDE.md
├── src/
│   ├── dataset1/News_dataset/{train_dataset.csv,test_dataset.csv}
│   ├── dataset2/News_dataset/{train_dataset.csv,test_dataset.csv}
│   └── Clickbait_dataset/{clickbait.csv}
└── app.py  (Flask API for inference)
```


***

## 3. Environment Setup

### A. Docker

Install Docker Desktop (Windows/Mac) or Docker Engine (Linux) plus NVIDIA Container Toolkit:

```bash
# Linux example
curl -fsSL https://get.docker.com | sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -fsSL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```


### B. Git LFS

```bash
git lfs install
```


***

## 4. Docker \& GPU Profiles

**Dockerfile** uses multi-layer caching:

1. Install system packages + editors (`nano`,`vim`)
2. `pip install --upgrade pip` once
3. Copy `requirements.txt`; install PyTorch+CUDA and pip deps
4. Copy code; create non-root user; set up Jupyter kernel

**docker-compose.dev.yml** ports and volumes:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"    # Flask
      - "8888:8888"    # Jupyter
    volumes:
      - ./:/app
      - ~/.cache/pip:/root/.cache/pip
      - pip_cache:/root/.cache/pip
    environment:
      - CUDA_VISIBLE_DEVICES=0   # change per GPU selection
volumes:
  pip_cache:
```

**GPU Profiles** (`--profile`):

- **low**: batch_size=8, max_length=128, epochs=3
- **high**: batch_size=32, max_length=512, epochs=5

***

## 5. Installing Dependencies

### Inside Docker

```bash
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up -d
```


### Locally

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


***

## 6. Git LFS for Large Files

Track datasets \& models:

```bash
git lfs track "src/dataset1/News_dataset/**"
git lfs track "src/dataset2/News_dataset/**"
git lfs track "src/Clickbait_dataset/**"
git add .gitattributes
git commit -m "Track data with LFS"
git add src/dataset1 src/dataset2 src/Clickbait_dataset
git commit -m "Add large CSVs via LFS"
git push
```


***

## 7. Training \& Validation

**Run training (first time)**

```bash
# inside container or local
python production_fake_news_detector.py --csv /app/src/dataset1/News_dataset/train_dataset.csv --profile low
```

- Performs 5-fold CV, class weighting, early stopping, gradient clipping.
- Saves best model under `production_model/best_model`.
- Generates `results/kfold_metrics.json` and `results/comprehensive_results.png`.

**Subsequent runs** reuse the model: omit `--csv` to load `CONFIG['default_csv']`.

***

## 8. Interactive Inference

After training:

```bash
# Launch interactive mode
python production_fake_news_detector.py --csv /app/src/dataset1/News_dataset/train_dataset.csv --profile low
# When prompted "Interactive? y/n": type y
```

Enter text; gets detailed output + logs session to `results/session_*.txt`.

***

## 9. Flask API Usage

**app.py** exposes endpoints:

- `GET /health`
- `POST /predict`
- `POST /batch_predict`
- `POST /detect_clickbait`
- `POST /batch_detect_clickbait`
- `POST /analyze_content`
- `GET /stats`

**Run API**:

```bash
python app.py
```

Access docs at `http://localhost:8000`.

***

## 10. Jupyter Notebook Access

Inside container:

```bash
# mount trainableModel folder
docker exec -it <container> bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app/src
```

Open `http://localhost:8888/?token=...` in Windows browser; edit .py files live.

***

## 11. Clickbait Detection Integration

Follow **CLICKBAIT-DETECTION.md** to add:

- `ClickbaitDetector` class
- `/detect_clickbait`, `/batch_detect_clickbait`, `/analyze_content` endpoints
- Tracking clickbait model via LFS

***

## 12. Troubleshooting

- **CUDA OOM**: lower batch_size / max_length
- **Missing CSV**: verify `--csv` path
- **Jupyter kernel**: ensure `--prefix` install by non-root
- **Docker cache**: omit `--build`; use pip cache volumes
- **User exists**: use `id -u appuser || useradd ...`

***

## 13. Quick Command Reference

| Task | Command |
| :-- | :-- |
| Build \& start | `docker-compose up -d` |
| Rebuild deps only | `docker-compose build` |
| Train + CV | `python production_fake_news_detector.py --csv <path> --profile low` |
| Interactive mode | `python production_fake_news_detector.py` + `y` |
| Flask API | `python app.py` |
| JupyterLab | `jupyter lab --ip=0.0.0.0 --port=8888` |
| LFS track data | `git lfs track "src/dataset*/**"` |
| View results | `ls results/` \| `cat results/session_*.txt` |


***

**Now you have a single, unified guide** to set up, train, serve, and develop on this Fake News \& Clickbait Detection project!

