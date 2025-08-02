
"""
Quick Start: Fake News Detection using Pre-trained RoBERTa Models
================================================================

This script demonstrates how to quickly detect fake news using pre-trained models
without any training. Just run this script and test with your own news articles!

Author: AI Research Assistant
Date: August 2025
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class QuickFakeNewsDetector:
    """
    Quick fake news detector using pre-trained models.
    """

    def __init__(self):
        """Initialize with available pre-trained models."""
        self.models = {
            'roberta_hamzab': 'hamzab/roberta-fake-news-classification',
            'roberta_winterforest': 'winterForestStump/Roberta-fake-news-detector',
            'roberta_jy': 'jy46604790/Fake-News-Bert-Detect',
            'distilroberta_vikram': 'vikram71198/distilroberta-base-finetuned-fake-news-detection'
        }
        self.current_model = None
        self.classifier = None

    def load_model(self, model_key='roberta_hamzab'):
        """
        Load a pre-trained model.

        Args:
            model_key (str): Key for the model to load
        """
        if model_key not in self.models:
            print(f"Available models: {list(self.models.keys())}")
            return

        model_name = self.models[model_key]
        print(f"Loading model: {model_name}")

        try:
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
            self.current_model = model_key
            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying with basic configuration...")

            # Fallback: load model manually
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            self.classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            self.current_model = model_key
            print("✅ Model loaded with fallback method!")

    def predict(self, text, format_text=True):
        """
        Predict if news is fake or real.

        Args:
            text (str): News text to analyze
            format_text (bool): Whether to format text for specific models

        Returns:
            dict: Prediction results
        """
        if not self.classifier:
            print("Please load a model first using load_model()")
            return None

        # Format text for specific models if needed
        if format_text and self.current_model == 'roberta_hamzab':
            # This model expects specific format
            if not text.startswith('<title>'):
                formatted_text = f"<title>{text[:100]}<content>{text}<end>"
            else:
                formatted_text = text
        else:
            formatted_text = text

        try:
            result = self.classifier(formatted_text)

            # Standardize output format
            if isinstance(result, list):
                result = result[0]

            # Map labels to standard format
            label = result['label'].upper()
            if 'FAKE' in label or label == 'LABEL_0':
                prediction = 'FAKE'
            else:
                prediction = 'REAL'

            confidence = result['score']

            return {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': prediction,
                'confidence': confidence,
                'raw_result': result
            }

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def test_sample_news(self):
        """Test with sample news articles."""
        sample_news = [
            {
                'text': "Scientists have discovered that drinking coffee prevents all diseases and makes you immortal. This revolutionary breakthrough will change humanity forever.",
                'expected': 'FAKE'
            },
            {
                'text': "The Federal Reserve announced today that interest rates will remain unchanged for the remainder of the quarter, citing concerns about inflation.",
                'expected': 'REAL'
            },
            {
                'text': "BREAKING: Aliens have landed in New York City and are demanding to speak to our planet's manager. They seem particularly upset about climate change.",
                'expected': 'FAKE'
            },
            {
                'text': "Local weather forecast shows partly cloudy skies tomorrow with temperatures reaching a high of 75 degrees Fahrenheit.",
                'expected': 'REAL'
            },
            {
                'text': "Miracle weight loss pill helps you lose 50 pounds overnight without diet or exercise! Doctors hate this one simple trick!",
                'expected': 'FAKE'
            }
        ]

        print("\n🧪 Testing Sample News Articles")
        print("=" * 50)

        correct_predictions = 0
        total_predictions = len(sample_news)

        for i, news in enumerate(sample_news, 1):
            print(f"\n📰 Test {i}:")
            print(f"Text: {news['text']}")
            print(f"Expected: {news['expected']}")

            result = self.predict(news['text'])
            if result:
                print(f"Predicted: {result['prediction']} (Confidence: {result['confidence']:.3f})")

                if result['prediction'] == news['expected']:
                    print("✅ Correct!")
                    correct_predictions += 1
                else:
                    print("❌ Incorrect")
            else:
                print("❌ Prediction failed")

        accuracy = correct_predictions / total_predictions
        print(f"\n📊 Test Results: {correct_predictions}/{total_predictions} correct ({accuracy:.1%})")

def main():
    """Main function to demonstrate quick fake news detection."""
    print("🚀 Quick Fake News Detection with Pre-trained RoBERTa")
    print("=" * 60)

    # Initialize detector
    detector = QuickFakeNewsDetector()

    # Try to load a model
    print("\n1️⃣ Loading pre-trained model...")
    detector.load_model('roberta_winterforest')  # Try winterforest first (smaller)

    if not detector.classifier:
        print("Trying alternative model...")
        detector.load_model('roberta_jy')

    if not detector.classifier:
        print("❌ Could not load any model. Please check your internet connection.")
        return

    # Test with sample news
    detector.test_sample_news()

    # Interactive testing
    print("\n\n2️⃣ Interactive Testing")
    print("=" * 30)
    print("Enter your own news articles to test (or 'quit' to exit):")

    while True:
        try:
            user_input = input("\n📝 Enter news text: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("Please enter some text.")
                continue

            result = detector.predict(user_input)
            if result:
                print(f"\n🔍 Analysis Result:")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.3f}")

                if result['prediction'] == 'FAKE':
                    print("⚠️  This appears to be fake news!")
                else:
                    print("✅ This appears to be legitimate news.")
            else:
                print("❌ Could not analyze the text.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
