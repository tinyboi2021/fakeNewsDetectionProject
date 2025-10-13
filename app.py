#!/usr/bin/env python3
"""
Content Analysis API using RoBERTa
A Flask-based web service for detecting fake news and clickbait using transformer models
"""

import os
import logging
from datetime import datetime
from typing import Dict, List
import torch
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-this')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 1048576))
CORS(app)

class FakeNewsDetector:
    """RoBERTa-based Fake News Detection Model"""

    def __init__(self, model_name: str = "hamzab/roberta-fake-news-classification"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained RoBERTa fake news classification model and tokenizer"""
        try:
            logger.info(f"Loading fake news model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Fake news model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading fake news model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict:
        """Predict if text is fake or real news"""
        try:
            if not text or not text.strip():
                return {
                    'error': 'Empty text provided',
                    'prediction': None,
                    'confidence': 0.0
                }
            text = text.strip()[:512]  # Limit for efficiency

            # Get prediction
            out = self.classifier(text)
            result = out[0]
            label = result['label'].lower()  # May be "LABEL_0"/"LABEL_1" or "real"/"fake"

            # Try to map to FAKE/REAL
            label_map = {
                "fake": "FAKE",
                "real": "REAL",
                "label_0": "REAL",  # Many models use 0:real, 1:fake, check model card on HF
                "label_1": "FAKE"
            }
            prediction = label_map.get(label, label.upper())
            confidence = float(result['score'])
            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_used": self.model_name,
                "raw_output": result,
                "text_length": len(text)
            }
        except Exception as e:
            logger.error(f"Fake news prediction error: {str(e)}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

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

# Initialize both detectors globally
logger.info("Initializing Fake News Detector...")
fake_news_detector = FakeNewsDetector(os.getenv('FAKE_NEWS_MODEL', 'hamzab/roberta-fake-news-classification'))

logger.info("Initializing Clickbait Detector...")
clickbait_detector = ClickbaitDetector(os.getenv('CLICKBAIT_MODEL', 'valurank/distilroberta-clickbait'))

# Helper functions for combined analysis
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

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'fake_news': fake_news_detector.model is not None,
            'clickbait': clickbait_detector.classifier is not None
        },
        'version': '2.0.0'
    })

# Fake News Detection Endpoints
@app.route('/predict', methods=['POST'])
def predict():
    """Single fake news prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request body'
            }), 400
        text = data['text']
        result = fake_news_detector.predict(text)
        if 'error' in result and result['error']:
            return jsonify(result), 500
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch fake news prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing texts field in request body'
            }), 400
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts field must be a list'
            }), 400
        if len(texts) > 10:
            return jsonify({
                'error': 'Maximum 10 texts allowed per batch'
            }), 400
        results = []
        for i, text in enumerate(texts):
            result = fake_news_detector.predict(text)
            results.append({
                'index': i,
                'text_preview': text[:100] + '...' if len(text) > 100 else text,
                'result': result
            })
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

# Clickbait Detection Endpoints
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

# Combined Analysis Endpoint
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

@app.route('/stats')
def stats():
    """API statistics endpoint"""
    return jsonify({
        'model_info': {
            'fake_news_model': fake_news_detector.model_name,
            'clickbait_model': clickbait_detector.model_name,
            'models_loaded': {
                'fake_news': fake_news_detector.model is not None,
                'clickbait': clickbait_detector.classifier is not None
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'system_info': {
            'python_version': os.sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        },
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle large request errors"""
    return jsonify({
        'error': 'Request too large',
        'max_size': app.config['MAX_CONTENT_LENGTH']
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('APP_PORT', 8000))
    debug = os.getenv('FLASK_ENV') == 'development'
    logger.info(f"Starting Content Analysis API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)