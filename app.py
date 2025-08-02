#!/usr/bin/env python3
"""
Fake News Detection API using RoBERTa
A Flask-based web service for detecting fake news using transformer models
"""

import os
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline
)
import numpy as np
import pandas as pd
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

# Enable CORS
CORS(app)

class FakeNewsDetector:
    """RoBERTa-based Fake News Detection Model"""

    def __init__(self, model_name: str = "roberta-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained RoBERTa model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")

            # For demo purposes, we'll use a general sentiment model
            # In production, you would use a model specifically fine-tuned for fake news
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Create pipeline
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )

            logger.info("Model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, text: str) -> Dict:
        """
        Predict if text is fake news

        Args:
            text (str): Input text to analyze

        Returns:
            Dict: Prediction results with confidence scores
        """
        try:
            if not text or not text.strip():
                return {
                    'error': 'Empty text provided',
                    'prediction': None,
                    'confidence': 0.0
                }

            # Clean and preprocess text
            text = text.strip()[:512]  # Limit to 512 characters for efficiency

            # Get prediction
            results = self.classifier(text)

            # Process results (this is a demo mapping)
            # In real implementation, you'd have a model trained specifically for fake news
            sentiment_scores = {item['label']: item['score'] for item in results[0]}

            # Demo logic: map sentiment to fake news probability
            # Negative sentiment might indicate more sensational/fake content
            fake_probability = sentiment_scores.get('LABEL_0', 0.5)  # Negative
            real_probability = sentiment_scores.get('LABEL_2', 0.5)  # Positive

            is_fake = fake_probability > real_probability
            confidence = max(fake_probability, real_probability)

            return {
                'prediction': 'FAKE' if is_fake else 'REAL',
                'confidence': float(confidence),
                'fake_probability': float(fake_probability),
                'real_probability': float(real_probability),
                'raw_scores': sentiment_scores,
                'text_length': len(text),
                'model_used': self.model_name
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

# Initialize the detector
logger.info("Initializing Fake News Detector...")
detector = FakeNewsDetector(os.getenv('MODEL_NAME', 'roberta-base'))

# Routes
@app.route('/')
def home():
    """Home page with API documentation"""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }
        code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; }
        .method { color: #007bff; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Fake News Detection API</h1>
        <p>A RoBERTa-based API for detecting fake news content.</p>

        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Analyze text for fake news detection</p>
            <p><strong>Request body:</strong></p>
            <code>{"text": "Your news article text here"}</code>

            <p><strong>Response:</strong></p>
            <code>
            {
              "prediction": "FAKE|REAL",
              "confidence": 0.85,
              "fake_probability": 0.85,
              "real_probability": 0.15
            }
            </code>
        </div>

        <div class="endpoint">
            <h3><span class="method">POST</span> /batch_predict</h3>
            <p>Analyze multiple texts at once</p>
            <p><strong>Request body:</strong></p>
            <code>{"texts": ["Text 1", "Text 2", "Text 3"]}</code>
        </div>

        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health status</p>
        </div>

        <div class="endpoint">
            <h3><span class="method">GET</span> /stats</h3>
            <p>Get API usage statistics</p>
        </div>
    </div>
</body>
</html>
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

        if 'error' in result:
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

        if len(texts) > 10:  # Limit batch size
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
    # Production deployment with Gunicorn is recommended
    # This is for development/testing only
    port = int(os.getenv('APP_PORT', 8000))
    debug = os.getenv('FLASK_ENV') == 'development'

    logger.info(f"Starting Fake News Detection API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
