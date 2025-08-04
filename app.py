#!/usr/bin/env python3
"""
Fake News Detection API using RoBERTa
A Flask-based web service for detecting fake news using a model trained for fake-news classification
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
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
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
            logger.error(f"Prediction error: {str(e)}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

# Initialize the detector globally
logger.info("Initializing Fake News Detector...")
detector = FakeNewsDetector(os.getenv('MODEL_NAME', 'hamzab/roberta-fake-news-classification'))

@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
    <h1>🔍 Fake News Detection API</h1>
    <p>A RoBERTa-based API for detecting fake news content.</p>

    <div style="background:#f8fafd;padding:20px;border-radius:5px;max-width:600px;">
      <b>POST /predict</b><br>
      <i>Request body:</i><br>
      <code>{"text": "Your news article text here"}</code>
      <br><i>Response:</i><br>
      <code>{ "prediction": "FAKE|REAL", "confidence": 0.95 }</code>
      <br><br>
      <b>POST /batch_predict</b><br>
      <i>Request body:</i><br>
      <code>{"texts": ["Text 1", "Text 2", "Text 3"]}</code>
      <br>
      <b>GET /health</b><br>
      <b>GET /stats</b><br>
    </div>
    """
    return render_template_string(html_template)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': detector.model is not None,
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single text prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request body'
            }), 400
        text = data['text']
        result = detector.predict(text)
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
    """Batch prediction endpoint"""
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
            result = detector.predict(text)
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

@app.route('/stats')
def stats():
    """API statistics endpoint"""
    return jsonify({
        'model_info': {
            'name': detector.model_name,
            'loaded': detector.model is not None,
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
    logger.info(f"Starting Fake News Detection API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
