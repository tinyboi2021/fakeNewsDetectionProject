# 🚀 Production Fake News Detector - User Guide

Complete guide for running the production-grade fake news detection system with K-fold validation and interactive interface.

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Prerequisites](#-prerequisites)
3. [Installation](#-installation)
4. [Configuration](#-configuration)
5. [Running With Training](#-running-with-training-first-time)
6. [Running Without Training](#-running-without-training-subsequent-runs)
7. [Interactive Mode](#-interactive-mode)
8. [Understanding Results](#-understanding-results)
9. [Troubleshooting](#-troubleshooting)
10. [Advanced Usage](#-advanced-usage)

---

## 🎯 Quick Start

### First Time (With Training)
```bash
# In Docker container
cd /app/trainableModel
python production_fake_news_detector.py
# Answer 'y' to enter interactive mode after training
```

### Subsequent Runs (Without Training)
```bash
# Model already trained, just run predictions
python production_fake_news_detector.py
# Answer 'y' for interactive mode
```

---

## 🔧 Prerequisites

### Required Software
- **Python 3.8+**
- **PyTorch with CUDA** (for GPU support)
- **Transformers library**
- **scikit-learn**
- **pandas, numpy**
- **matplotlib, seaborn**

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB+ |
| **GPU VRAM** | 6GB (RTX 4060 Ti) | 16GB+ (A100) |
| **Storage** | 10GB | 20GB+ |
| **CPU** | 4 cores | 8+ cores |

### Dataset Requirements
- **Fake news CSV**: `/app/src/dataset1/News_dataset/Fake.csv`
- **Real news CSV**: `/app/src/dataset1/News_dataset/True.csv`

**CSV Format:**
```csv
title,text,subject,date
"News title here","News content here","politics","2023-01-01"
```

---

## 📦 Installation

### Inside Docker Container (Recommended)

```bash
# 1. Access your Docker container
docker exec -it fake_news_app_dev bash

# 2. Navigate to training directory
cd /app/trainableModel

# 3. Verify dependencies
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 4. Copy the script
# (production_fake_news_detector.py should already be in trainableModel/)

# 5. Ready to run!
python production_fake_news_detector.py
```

### Local Installation (Without Docker)

```bash
# 1. Clone repository
git clone <your-repo>
cd fake-news-detection

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers scikit-learn pandas numpy matplotlib seaborn

# 3. Prepare dataset
# Place Fake.csv and True.csv in appropriate folders

# 4. Update paths in script
# Edit CONFIG dictionary to point to your dataset location

# 5. Run
python production_fake_news_detector.py
```

---

## ⚙️ Configuration

The script uses a `CONFIG` dictionary for all settings. You can modify it before running:

```python
CONFIG = {
    'model_name': 'roberta-base',          # Pre-trained model
    'max_length': 256,                     # Max token length
    'batch_size': 16,                      # Batch size for training
    'learning_rate': 2e-5,                 # Learning rate
    'num_epochs': 4,                       # Epochs per fold
    'n_folds': 5,                          # K-fold splits
    'warmup_ratio': 0.1,                   # LR warmup ratio
    'weight_decay': 0.01,                  # Weight decay
    'max_grad_norm': 1.0,                  # Gradient clipping
    'early_stopping_patience': 2,          # Early stop patience
    'random_seed': 42,                     # Random seed
    'model_dir': './production_model',     # Save directory
    'results_dir': './results',            # Results directory
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### Hardware-Specific Profiles

**For RTX 4060 Ti (6GB VRAM):**
```python
CONFIG = {
    'batch_size': 8,      # Smaller batch
    'max_length': 128,    # Shorter sequences
    'num_epochs': 3,      # Fewer epochs
    # ... rest of config
}
```

**For A100 or High-End GPU (40GB+ VRAM):**
```python
CONFIG = {
    'batch_size': 32,     # Larger batch
    'max_length': 512,    # Full sequences
    'num_epochs': 5,      # More epochs
    # ... rest of config
}
```

---

## 🏋️ Running With Training (First Time)

### Step 1: Start the Program

```bash
python production_fake_news_detector.py
```

### Step 2: Training Process

The program will automatically:

1. **Check for existing model**
   ```
   📚 No pre-trained model found. Starting training...
   ```

2. **Load dataset**
   ```
   📂 Loading data...
   ✅ Loaded 44898 articles
      - Real news: 21417 (47.7%)
      - Fake news: 23481 (52.3%)
   ```

3. **Calculate class weights**
   ```
   ⚖️  Class weights: Fake=0.96, Real=1.05
   ```

4. **Train 5 folds** (this takes time!)
   ```
   ============================================================
   Training Fold 1/5
   ============================================================
   Epoch 1/4 | Batch 50/2245 | Loss: 0.3241
   ...
   📊 Epoch 1 Results:
      Train Loss: 0.2891 | Train Acc: 0.8923 | Train F1: 0.8915
      Val Loss: 0.2145 | Val Acc: 0.9234 | Val F1: 0.9228 | Val AUC: 0.9567
   ```

5. **Show fold results**
   ```
   ✅ Fold 1 Results:
      Accuracy: 0.9347
      F1-Score: 0.9342
      ROC-AUC: 0.9723
      Brier Score: 0.0524
   ```

6. **Display average metrics**
   ```
   ============================================================
   📊 CROSS-VALIDATION RESULTS (Average across 5 folds)
   ============================================================
   Accuracy:    0.9312 ± 0.0045
   F1-Score:    0.9308 ± 0.0048
   ROC-AUC:     0.9701 ± 0.0032
   Brier Score: 0.0551 ± 0.0023
   ============================================================
   ```

7. **Create visualizations**
   ```
   📊 Visualizations saved to ./results/comprehensive_results.png
   ```

8. **Save best model**
   ```
   💾 Best model (Fold 3) saved to ./production_model/best_model
      Accuracy: 0.9367
   ```

### Step 3: Enter Interactive Mode

```
============================================================
Enter interactive mode? (y/n): y
```

**Type 'y' to start testing predictions immediately!**

### Expected Training Time

| Hardware | Training Time (5 folds × 4 epochs) |
|----------|-----------------------------------|
| **RTX 4060 Ti** | ~2-3 hours |
| **RTX 3090** | ~1.5-2 hours |
| **A100** | ~45-60 minutes |
| **CPU Only** | ~10-15 hours (not recommended) |

---

## ✅ Running Without Training (Subsequent Runs)

Once trained, the model is saved and can be reused instantly!

### Step 1: Run the Script

```bash
python production_fake_news_detector.py
```

### Step 2: Model Loads Automatically

```
============================================================
🚀 PRODUCTION-GRADE FAKE NEWS DETECTION SYSTEM
============================================================
Combining K-fold validation with interactive interface
============================================================
🚀 Initialized Production Fake News Detector
📱 Device: cuda
🤖 Model: roberta-base

✅ Pre-trained model found. Skipping training.
```

### Step 3: Choose Interactive Mode

```
============================================================
Enter interactive mode? (y/n): y
```

### Step 4: Start Making Predictions!

```
============================================================
🎯 INTERACTIVE FAKE NEWS DETECTION
============================================================
Enter news text to analyze. Type 'exit' to quit.
------------------------------------------------------------

📝 Enter news text (or 'exit'): 
```

---

## 🎮 Interactive Mode

### Using the Interactive Interface

1. **Enter text to analyze:**
   ```
   📝 Enter news text (or 'exit'): SHOCKING: Scientists discover cure for aging!
   ```

2. **View detailed results:**
   ```
   🤔 Analyzing...

   ============================================================
   🔍 PREDICTION RESULTS
   ============================================================
   📝 Input: SHOCKING: Scientists discover cure for aging!
   🎯 Prediction: Fake
   🔢 Confidence: 0.9856 (98.56%)
   🚨 Fake Probability: 0.9856
   ✅ Real Probability: 0.0144
   📊 Confidence Level: 🟢 Very High
   ============================================================
   💾 Results saved to: ./results/prediction_session_20251016_195430.txt
   ```

3. **Test multiple articles:**
   - Each prediction is logged automatically
   - Session file tracks all predictions with timestamps
   - Continue testing as many articles as you want

4. **Exit when done:**
   ```
   📝 Enter news text (or 'exit'): exit

   👋 Exiting interactive mode.
   📁 Session log: ./results/prediction_session_20251016_195430.txt
   📊 Total predictions: 15
   ```

### Example Predictions

**Real News Example:**
```
Input: "NASA announces successful Mars rover landing with new scientific instruments"
Prediction: Real
Confidence: 94.23%
```

**Fake News Example:**
```
Input: "Miracle pill cures all diseases, doctors don't want you to know!"
Prediction: Fake
Confidence: 99.12%
```

---

## 📊 Understanding Results

### Training Metrics Explained

| Metric | What It Means | Good Score |
|--------|---------------|------------|
| **Accuracy** | Overall correct predictions | >90% |
| **F1-Score** | Balance of precision/recall | >0.90 |
| **ROC-AUC** | Ranking quality | >0.95 |
| **Brier Score** | Probability calibration | <0.10 (lower is better) |

### Confidence Levels

| Confidence | Visual Indicator | Meaning |
|------------|------------------|---------|
| **≥90%** | 🟢 Very High | Trust this prediction |
| **70-90%** | 🟡 High | Fairly reliable |
| **50-70%** | 🟠 Moderate | Verify if possible |
| **<50%** | 🔴 Low | Model uncertain |

### Output Files

**1. Model Files** (`./production_model/`)
```
best_model/
├── config.json           # Model configuration
├── pytorch_model.bin     # Trained weights
├── tokenizer_config.json # Tokenizer settings
└── vocab.json           # Vocabulary

fold_0_best.pt           # Checkpoint for fold 0
fold_1_best.pt           # Checkpoint for fold 1
...
```

**2. Results Files** (`./results/`)
```
comprehensive_results.png         # Visualizations
kfold_metrics.json               # Training metrics
prediction_session_20251016.txt  # Prediction logs
```

### Visualization Guide

**Confusion Matrix:**
- Shows true vs predicted labels
- Diagonal = correct predictions
- Off-diagonal = errors

**ROC Curve:**
- Higher curve = better model
- Area under curve (AUC) should be >0.95

**Calibration Curve:**
- Closer to diagonal = better calibrated
- Good calibration means confidence scores are accurate

**Metrics Comparison:**
- Shows consistency across folds
- Similar bars = stable model

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size in CONFIG
CONFIG['batch_size'] = 8  # or even 4
CONFIG['max_length'] = 128
```

#### 2. Dataset Not Found

**Error:**
```
FileNotFoundError: No such file or directory: '/app/src/dataset1/News_dataset/Fake.csv'
```

**Solution:**
```bash
# Check file paths
ls /app/src/dataset1/News_dataset/

# Update paths in main() function if needed
df = detector.load_data(
    fake_csv_path="/your/actual/path/Fake.csv",
    true_csv_path="/your/actual/path/True.csv"
)
```

#### 3. Model Not Loading

**Error:**
```
FileNotFoundError: No saved model found at ./production_model/best_model
```

**Solution:**
```bash
# Train the model first
python production_fake_news_detector.py
# Answer 'y' when asked to enter interactive mode

# Or check if model directory exists
ls ./production_model/
```

#### 4. Slow Training on CPU

**Issue:** Training taking >10 hours

**Solution:**
```bash
# Enable GPU if available
nvidia-smi  # Check GPU availability

# Or reduce dataset size for testing
# Edit load_data() to sample data:
df = df.sample(n=5000, random_state=42)
```

#### 5. Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: './production_model'
```

**Solution:**
```bash
# Create directories manually
mkdir -p production_model results

# Or run with proper permissions
docker exec -u 0 -it fake_news_app_dev chown -R appuser:appuser /app
```

---

## 🎓 Advanced Usage

### 1. Customize Hyperparameters

Edit the CONFIG dictionary before running:

```python
# For faster testing (lower accuracy)
CONFIG['n_folds'] = 3
CONFIG['num_epochs'] = 2

# For maximum accuracy (slower)
CONFIG['n_folds'] = 10
CONFIG['num_epochs'] = 6
CONFIG['batch_size'] = 32
```

### 2. Use Pre-trained Fine-tuned Model

```python
# Instead of 'roberta-base', use a fake news model
CONFIG['model_name'] = 'hamzab/roberta-fake-news-classification'
```

### 3. Export Predictions Programmatically

```python
# Load model and make batch predictions
detector = ProductionFakeNewsDetector(CONFIG)
model, tokenizer = detector.load_best_model()

texts = [
    "News article 1...",
    "News article 2...",
    "News article 3..."
]

for text in texts:
    result = detector.predict_single(text, model, tokenizer)
    print(f"{text[:50]}: {result['prediction']} ({result['confidence']:.2%})")
```

### 4. Integrate with API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
detector = ProductionFakeNewsDetector(CONFIG)
model, tokenizer = detector.load_best_model()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    result = detector.predict_single(text, model, tokenizer)
    return jsonify(result)

app.run(port=5000)
```

### 5. Compare Multiple Models

```python
models_to_compare = [
    'roberta-base',
    'roberta-large',
    'bert-base-uncased'
]

for model_name in models_to_compare:
    CONFIG['model_name'] = model_name
    detector = ProductionFakeNewsDetector(CONFIG)
    # Train and compare results
```

---

## 📝 Best Practices

### For Training
1. ✅ **Always use GPU** for reasonable training times
2. ✅ **Start with default config** before optimizing
3. ✅ **Monitor GPU memory** usage during training
4. ✅ **Save training logs** for later analysis
5. ✅ **Use validation metrics** to avoid overfitting

### For Production Use
1. ✅ **Load model once** and reuse for predictions
2. ✅ **Batch predictions** when possible for efficiency
3. ✅ **Log all predictions** for audit trails
4. ✅ **Set confidence thresholds** based on your use case
5. ✅ **Regularly retrain** with new data

---

## 🎯 Quick Reference

### Key Commands

```bash
# Train from scratch
python production_fake_news_detector.py

# Interactive mode (after training)
python production_fake_news_detector.py
# Type 'y' when prompted

# Check GPU status
nvidia-smi

# Monitor training
watch -n1 nvidia-smi

# View results
ls -lh ./results/
cat ./results/prediction_session_*.txt
```

### File Locations

| File Type | Location | Purpose |
|-----------|----------|---------|
| **Script** | `/app/trainableModel/production_fake_news_detector.py` | Main program |
| **Dataset** | `/app/src/dataset1/News_dataset/` | Training data |
| **Model** | `./production_model/best_model/` | Trained model |
| **Results** | `./results/` | Metrics & predictions |
| **Logs** | `./results/prediction_session_*.txt` | Session logs |

---

## 🆘 Getting Help

### If you encounter issues:

1. **Check prerequisites** - ensure all dependencies installed
2. **Verify GPU** - run `nvidia-smi` to check availability
3. **Review logs** - check error messages carefully
4. **Reduce complexity** - try smaller batch size or fewer folds
5. **Test dataset** - verify CSV files are properly formatted

### Support Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Review inline code comments
- **Community**: Ask questions in project discussions

---

## 🎉 Summary

| Task | Command | Time |
|------|---------|------|
| **First run (training)** | `python production_fake_news_detector.py` | 1-3 hours |
| **Subsequent runs** | `python production_fake_news_detector.py` | <1 minute |
| **Interactive testing** | Answer 'y' when prompted | Instant |
| **Batch predictions** | Use `predict_single()` in code | ~0.1s per text |

**Expected Accuracy:** 93-96%

**Now you're ready to detect fake news with production-grade accuracy!** 🚀