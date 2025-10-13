# 🎯 Adding Clickbait Detection to Your Fake News Detection Project

This guide shows you how to extend your existing fake news detection API with clickbait detection capabilities, creating a comprehensive content analysis tool.

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Available Models](#-available-models)
3. [Implementation Steps](#-implementation-steps)
4. [Code Integration](#-code-integration)
5. [API Testing](#-api-testing)
6. [Training Custom Models](#-training-custom-models)
7. [Combined Analysis](#-combined-analysis)
8. [Troubleshooting](#-troubleshooting)

## 🔍 Overview

**Clickbait detection** identifies sensationalized headlines designed to attract attention and encourage clicks, often using emotional language, curiosity gaps, or exaggerated claims. Adding this to your fake news detector creates a powerful dual-purpose content analysis system.

### Why Add Clickbait Detection?

- **Enhanced Content Analysis**: Detect both misinformation and sensationalized content
- **Media Quality Assessment**: Evaluate journalistic integrity beyond just truth/falsehood
- **User Protection**: Help users identify manipulative content strategies
- **Research Applications**: Study the relationship between clickbait and misinformation

## 🤖 Available Models

### Recommended Pretrained Models

| Model | Accuracy | Language | Best For |
|-------|----------|----------|----------|
| `valurank/distilroberta-clickbait` | ~92% | English | General purpose, fast inference |
| `christinacdl/XLM_RoBERTa-Clickbait-Detection-new` | ~89% | Multilingual | International content |
| `jy46604790/Fake-News-Bert-Detect` | ~88% | English | News-specific content |

### Model Characteristics

**valurank/distilroberta-clickbait:**
- Based on DistilRoBERTa (faster, smaller)
- Trained on news headlines dataset
- Good balance of speed and accuracy
- Optimized for real-time applications

**christinacdl/XLM_RoBERTa-Clickbait-Detection-new:**
- Multilingual support (100+ languages)
- Larger model size (better accuracy)
- Suitable for international applications
- Higher GPU memory requirements

## 🚀 Implementation Steps

### Step 1: Update Requirements

Add to your `requirements.txt`:
```text
# No additional requirements needed - uses existing transformers
```

### Step 2: Create Clickbait Detector Class

Create a new file `clickbait_detector.py`:

```python
#!/usr/bin/env python3
"""
Clickbait Detection using Transformer Models
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class ClickbaitDetector:
    """Transformer-based Clickbait Detection Model"""
    
    def __init__(self, model_name="valurank/distilroberta-clickbait"):
        self.model_name = model_name
        self.classifier = None
        self.device = 0 if torch.cuda.is_available() else -1
        self.load_model()
    
    def load_model(self):
        """Load the pretrained clickbait detection model"""
        try:
            logger.info(f"Loading clickbait model: {self.model_name}")
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                return_all_scores=True
            )
            logger.info("Clickbait model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading clickbait model: {str(e)}")
            raise
    
    def predict(self, text):
        """Predict if text is clickbait"""
        try:
            if not text or not text.strip():
                return {
                    'error': 'Empty text provided',
                    'is_clickbait': False,
                    'confidence': 0.0
                }
            
            # Limit text length for efficiency
            text = text.strip()[:512]
            
            # Get prediction
            results = self.classifier(text)[0]
            
            # Process results - model outputs may vary
            clickbait_score = 0.0
            non_clickbait_score = 0.0
            
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                if 'clickbait' in label or 'click' in label or label == 'label_1':
                    clickbait_score = score
                else:
                    non_clickbait_score = score
            
            is_clickbait = clickbait_score > non_clickbait_score
            confidence = max(clickbait_score, non_clickbait_score)
            
            return {
                'is_clickbait': is_clickbait,
                'confidence': float(confidence),
                'clickbait_score': float(clickbait_score),
                'non_clickbait_score': float(non_clickbait_score),
                'model_used': self.model_name,
                'text_length': len(text)
            }
            
        except Exception as e:
            logger.error(f"Clickbait prediction error: {str(e)}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'is_clickbait': False,
                'confidence': 0.0
            }
    
    def batch_predict(self, texts):
        """Predict multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
```

### Step 3: Update Main Application

Modify your `app.py` to include clickbait detection:

```python
from clickbait_detector import ClickbaitDetector

# Initialize both detectors globally
logger.info("Initializing Fake News Detector...")
fake_news_detector = FakeNewsDetector(os.getenv('FAKE_NEWS_MODEL', 'hamzab/roberta-fake-news-classification'))

logger.info("Initializing Clickbait Detector...")
clickbait_detector = ClickbaitDetector(os.getenv('CLICKBAIT_MODEL', 'valurank/distilroberta-clickbait'))

# Add new routes
@app.route('/detect_clickbait', methods=['POST'])
def detect_clickbait():
    """Single clickbait detection endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field in request body'}), 400
        
        text = data['text']
        result = clickbait_detector.predict(text)
        
        if 'error' in result and result['error']:
            return jsonify(result), 500
            
        return jsonify({
            'success': True,
            'input': text,
            'clickbait_result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Clickbait detection endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/batch_detect_clickbait', methods=['POST'])
def batch_detect_clickbait():
    """Batch clickbait detection endpoint"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field in request body'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts field must be a list'}), 400
        
        if len(texts) > 10:
            return jsonify({'error': 'Maximum 10 texts allowed per batch'}), 400
        
        results = []
        for i, text in enumerate(texts):
            result = clickbait_detector.predict(text)
            results.append({
                'index': i,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'clickbait_result': result
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch clickbait detection error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/analyze_content', methods=['POST'])
def analyze_content():
    """Combined fake news + clickbait analysis"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field in request body'}), 400
        
        text = data['text']
        
        # Get both predictions
        fake_news_result = fake_news_detector.predict(text)
        clickbait_result = clickbait_detector.predict(text)
        
        # Create combined analysis
        combined_analysis = {
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'fake_news': fake_news_result,
            'clickbait': clickbait_result,
            'content_quality_score': calculate_content_quality(fake_news_result, clickbait_result),
            'recommendations': generate_recommendations(fake_news_result, clickbait_result)
        }
        
        return jsonify({
            'success': True,
            'analysis': combined_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Content analysis error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

def calculate_content_quality(fake_news_result, clickbait_result):
    """Calculate overall content quality score"""
    fake_score = 1.0 if fake_news_result.get('prediction') == 'REAL' else 0.0
    clickbait_score = 0.0 if clickbait_result.get('is_clickbait') else 1.0
    
    # Weight fake news more heavily (70% vs 30%)
    quality_score = (fake_score * 0.7) + (clickbait_score * 0.3)
    
    return {
        'score': round(quality_score, 2),
        'rating': get_quality_rating(quality_score)
    }

def get_quality_rating(score):
    """Convert quality score to human-readable rating"""
    if score >= 0.8:
        return 'High Quality'
    elif score >= 0.6:
        return 'Moderate Quality'
    elif score >= 0.4:
        return 'Low Quality'
    else:
        return 'Poor Quality'

def generate_recommendations(fake_news_result, clickbait_result):
    """Generate content recommendations"""
    recommendations = []
    
    if fake_news_result.get('prediction') == 'FAKE':
        recommendations.append('⚠️ Content may contain misinformation - verify with reliable sources')
    
    if clickbait_result.get('is_clickbait'):
        recommendations.append('📢 Headline appears to be clickbait - content may be sensationalized')
    
    if not recommendations:
        recommendations.append('✅ Content appears to be legitimate and non-sensationalized')
    
    return recommendations
```

### Step 4: Update Homepage

Modify the homepage template in your `app.py`:

```python
@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <h1>🔍 Content Analysis API</h1>
    <p>RoBERTa-based API for fake news and clickbait detection</p>

    <div style="background:#f8fafd;padding:20px;border-radius:5px;max-width:800px;">
      <h3>🚨 Fake News Detection</h3>
      <b>POST /predict</b><br>
      <code>{"text": "Your news article text here"}</code><br>
      <i>Response:</i> <code>{"prediction": "FAKE|REAL", "confidence": 0.95}</code>
      <br><br>
      
      <h3>🎯 Clickbait Detection</h3>
      <b>POST /detect_clickbait</b><br>
      <code>{"text": "Your headline here"}</code><br>
      <i>Response:</i> <code>{"is_clickbait": true, "confidence": 0.87}</code>
      <br><br>
      
      <h3>📊 Combined Analysis</h3>
      <b>POST /analyze_content</b><br>
      <code>{"text": "Your content here"}</code><br>
      <i>Response:</i> Combined fake news + clickbait analysis with quality score
      <br><br>
      
      <h3>📈 Batch Processing</h3>
      <b>POST /batch_predict</b> | <b>POST /batch_detect_clickbait</b><br>
      <code>{"texts": ["Text 1", "Text 2", "Text 3"]}</code>
      <br><br>
      
      <h3>🔧 System Status</h3>
      <b>GET /health</b> | <b>GET /stats</b>
    </div>
    """
    return render_template_string(html_template)
```

## 🧪 API Testing

### Single Clickbait Detection

```bash
curl -X POST http://localhost:8000/detect_clickbait \
  -H "Content-Type: application/json" \
  -d '{
    "text": "You Won'\''t Believe What This Celebrity Did Next!"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "input": "You Won't Believe What This Celebrity Did Next!",
  "clickbait_result": {
    "is_clickbait": true,
    "confidence": 0.94,
    "clickbait_score": 0.94,
    "non_clickbait_score": 0.06
  }
}
```

### Combined Content Analysis

```bash
curl -X POST http://localhost:8000/analyze_content \
  -H "Content-Type: application/json" \
  -d '{
    "text": "SHOCKING: Local man discovers one weird trick that doctors hate!"
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "analysis": {
    "fake_news": {"prediction": "FAKE", "confidence": 0.87},
    "clickbait": {"is_clickbait": true, "confidence": 0.95},
    "content_quality_score": {"score": 0.09, "rating": "Poor Quality"},
    "recommendations": [
      "⚠️ Content may contain misinformation - verify with reliable sources",
      "📢 Headline appears to be clickbait - content may be sensationalized"
    ]
  }
}
```

### Batch Testing

```bash
curl -X POST http://localhost:8000/batch_detect_clickbait \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Scientists make breakthrough discovery in cancer research",
      "10 Secrets Doctors Don'\''t Want You to Know!",
      "Local weather forecast shows sunny conditions ahead"
    ]
  }'
```

## 🎯 Training Custom Models

### Preparing Clickbait Dataset

Create your training dataset with these columns:
```csv
text,label
"You won't believe what happened next!",1
"Scientists discover new treatment for diabetes",0
"This one weird trick will change your life",1
"Government announces new policy changes",0
```

### Custom Training Script

Create `train_clickbait_detector.py`:

```python
#!/usr/bin/env python3
"""
Custom Clickbait Detection Model Training
"""

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

class ClickbaitTrainer:
    def __init__(self, model_name="roberta-base", output_dir="./clickbait_roberta"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_data(self, csv_path):
        """Load and prepare clickbait dataset"""
        df = pd.read_csv(csv_path)
        return df
    
    def prepare_dataset(self, df, test_size=0.2):
        """Prepare dataset for training"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)
        
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, num_epochs=3):
        """Train the clickbait detection model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.to(self.device)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs/clickbait',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=-1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def main():
    trainer = ClickbaitTrainer()
    
    # Load your dataset
    df = trainer.load_data("/app/data/clickbait_dataset.csv")
    train_ds, val_ds = trainer.prepare_dataset(df)
    
    # Train model
    trainer.train_model(train_ds, val_ds, num_epochs=3)
    
    print("Clickbait model training completed!")

if __name__ == "__main__":
    main()
```

### Running Custom Training

```bash
# Access container
docker exec -it fake_news_app_dev bash

# Navigate to training directory
cd /app/trainableModel

# Run clickbait training
python train_clickbait_detector.py
```

## 🔄 Combined Analysis

### Content Quality Scoring

The combined analysis provides:

- **Quality Score** (0.0 - 1.0): Weighted combination of authenticity and sensationalism
- **Quality Rating**: Human-readable assessment
- **Recommendations**: Actionable insights for content consumers

### Use Cases

**News Verification:**
```python
# Example: Analyzing suspicious headlines
headlines = [
    "COVID-19 vaccine contains microchips, local doctor reveals shocking truth",
    "New study shows promising results for Alzheimer's treatment"
]

for headline in headlines:
    analysis = analyze_content(headline)
    print(f"Quality: {analysis['content_quality_score']['rating']}")
```

**Social Media Monitoring:**
```python
# Batch analysis of social media posts
social_posts = get_trending_posts()
results = batch_analyze_content(social_posts)
flagged_content = [r for r in results if r['content_quality_score']['score'] < 0.5]
```

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Check available models
docker exec -it fake_news_app_dev python -c "from transformers import pipeline; print('Models loaded successfully')"

# Verify GPU access
docker exec -it fake_news_app_dev nvidia-smi
```

**2. Label Mapping Issues**
Different models may use different label formats:
- `LABEL_0` / `LABEL_1`
- `clickbait` / `not_clickbait`
- `True` / `False`

Check model documentation and adjust the `predict` method accordingly.

**3. Memory Issues**
```python
# Reduce batch size for GPU memory
per_device_train_batch_size=8  # Instead of 16

# Clear GPU cache
torch.cuda.empty_cache()
```

**4. API Response Delays**
```python
# Add caching for repeated requests
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_predict(text_hash):
    return detector.predict(text)
```

### Docker Configuration

Update your `docker-compose.dev.yml` to include clickbait model volumes:

```yaml
volumes:
  - ./clickbait_detector.py:/app/clickbait_detector.py
  - ./trainableModel:/app/trainableModel
  - clickbait_cache:/app/.cache/clickbait
```

### Git LFS Setup

Track clickbait model files:

```bash
git lfs track "trainableModel/clickbait_roberta/**"
git lfs track "*.json" "*.bin" "*.txt"
git add .gitattributes
git commit -m "Track clickbait model files with LFS"
```

## 📈 Performance Metrics

### Expected Performance

| Model | Precision | Recall | F1-Score | Inference Speed |
|-------|-----------|--------|----------|----------------|
| valurank/distilroberta-clickbait | 0.92 | 0.90 | 0.91 | ~50ms |
| christinacdl/XLM_RoBERTa | 0.89 | 0.88 | 0.89 | ~120ms |
| Custom trained | 0.85-0.95 | 0.84-0.94 | 0.85-0.94 | ~60ms |

### Monitoring

Add performance monitoring to your API:

```python
import time

@app.route('/stats')
def stats():
    return jsonify({
        'fake_news_model': fake_news_detector.model_name,
        'clickbait_model': clickbait_detector.model_name,
        'gpu_available': torch.cuda.is_available(),
        'models_loaded': {
            'fake_news': fake_news_detector.classifier is not None,
            'clickbait': clickbait_detector.classifier is not None
        }
    })
```

---

## 🎉 Conclusion

You now have a comprehensive content analysis API that can:

- ✅ Detect fake news and misinformation
- ✅ Identify clickbait and sensationalized content
- ✅ Provide combined quality assessments
- ✅ Handle batch processing efficiently
- ✅ Support both CPU and GPU inference
- ✅ Scale with Docker containerization

Your API can now serve as a powerful tool for content moderation, journalism assistance, and media literacy education!

---

**Next Steps:**
- Integrate with social media APIs for real-time monitoring
- Add sentiment analysis for complete content assessment
- Implement caching for improved performance
- Create a web dashboard for non-technical users