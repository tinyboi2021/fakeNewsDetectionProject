
"""
Fake News Detection using RoBERTa Model
=====================================

This script implements a complete fake news detection system using RoBERTa (Robustly Optimized BERT Pretraining Approach).
The model achieves high accuracy in classifying news articles as fake or real.

Requirements:
- transformers
- torch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- datasets

Author: AI Research Assistant
Date: August 2025
"""

import torch
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    pipeline
)
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FakeNewsDetector:
    """
    A comprehensive fake news detection system using RoBERTa model.
    """

    def __init__(self, model_name="roberta-base"):
        """
        Initialize the fake news detector.

        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def preprocess_text(self, text):
        """
        Clean and preprocess text data.

        Args:
            text (str): Raw text to preprocess

        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def load_data(self, fake_csv_path=None, true_csv_path=None):
        """
        Load and prepare the fake news dataset.

        Args:
            fake_csv_path (str): Path to fake news CSV file
            true_csv_path (str): Path to true news CSV file

        Returns:
            pd.DataFrame: Combined and preprocessed dataset
        """
        if fake_csv_path and true_csv_path:
            # Load from separate files
            fake_df = pd.read_csv(fake_csv_path)
            true_df = pd.read_csv(true_csv_path)

            fake_df['label'] = 0  # Fake news
            true_df['label'] = 1  # Real news

            # Combine datasets
            df = pd.concat([fake_df, true_df], ignore_index=True)
        else:
            # Create sample data for demonstration
            print("No dataset provided. Creating sample data for demonstration...")
            sample_data = {
                'title': [
                    'Breaking: Scientists discover cure for all diseases',
                    'Local weather forecast shows sunny skies ahead',
                    'Aliens land in Times Square, demand pizza',
                    'Stock market opens with moderate gains',
                    'Miracle diet makes you lose 50 pounds overnight',
                    'University announces new scholarship program'
                ],
                'text': [
                    'Revolutionary breakthrough in medical science promises to end all human suffering.',
                    'Tomorrow will be partly cloudy with temperatures reaching 75 degrees.',
                    'Extraterrestrial visitors reportedly very hungry after long journey from space.',
                    'Market analysts predict steady growth in technology sector.',
                    'Amazing weight loss secret that doctors hate! Click here to learn more.',
                    'The scholarship program will provide financial aid to deserving students.'
                ],
                'label': [0, 1, 0, 1, 0, 1]  # 0 = fake, 1 = real
            }
            df = pd.DataFrame(sample_data)

        # Combine title and text
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

        # Preprocess text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)

        # Remove empty texts
        df = df[df['processed_text'].str.len() > 10]

        print(f"Dataset loaded: {len(df)} articles")
        print(f"Real news: {sum(df['label'] == 1)}")
        print(f"Fake news: {sum(df['label'] == 0)}")

        return df

    def prepare_dataset(self, df, test_size=0.2, max_length=512):
        """
        Prepare dataset for training.

        Args:
            df (pd.DataFrame): Preprocessed dataframe
            test_size (float): Proportion of test data
            max_length (int): Maximum sequence length for tokenization

        Returns:
            tuple: Training and validation datasets
        """
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['processed_text'].tolist(),
            df['label'].tolist(),
            test_size=test_size,
            random_state=42,
            stratify=df['label']
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize data
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Create datasets
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

    def train_model(self, train_dataset, val_dataset, output_dir="./fake_news_roberta", 
                   num_epochs=3, learning_rate=2e-5, batch_size=16):
        """
        Train the RoBERTa model for fake news detection.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir (str): Directory to save the model
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size for training
        """
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        # Move model to device
        self.model.to(self.device)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            seed=42,
        )

        # Define compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train model
        print("Starting training...")
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")

        return trainer

    def evaluate_model(self, val_dataset, trainer=None):
        """
        Evaluate the trained model.

        Args:
            val_dataset: Validation dataset
            trainer: Trained model trainer

        Returns:
            dict: Evaluation metrics
        """
        if trainer:
            # Evaluate using trainer
            results = trainer.evaluate()
            print("\nEvaluation Results:")
            for key, value in results.items():
                print(f"{key}: {value:.4f}")

            # Get predictions for detailed analysis
            predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids

        else:
            # Evaluate using loaded model
            if not self.model:
                raise ValueError("No model loaded. Train a model first or load a pre-trained one.")

            # Get predictions
            y_pred = []
            y_true = val_dataset['labels']

            self.model.eval()
            with torch.no_grad():
                for i in range(len(val_dataset)):
                    input_ids = val_dataset[i]['input_ids'].unsqueeze(0).to(self.device)
                    attention_mask = val_dataset[i]['attention_mask'].unsqueeze(0).to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
                    y_pred.append(prediction)

        # Calculate detailed metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'])
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true
        }

    def load_trained_model(self, model_path):
        """
        Load a previously trained model.

        Args:
            model_path (str): Path to the saved model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")

    def predict(self, text):
        """
        Predict whether a single text is fake or real news.

        Args:
            text (str): News text to classify

        Returns:
            dict: Prediction results with confidence scores
        """
        if not self.model or not self.tokenizer:
            raise ValueError("No model loaded. Train a model first or load a pre-trained one.")

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        # Move to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get results
        fake_prob = probabilities[0][0].item()
        real_prob = probabilities[0][1].item()
        predicted_label = "Real" if real_prob > fake_prob else "Fake"
        confidence = max(fake_prob, real_prob)

        return {
            'text': text,
            'prediction': predicted_label,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob
        }

    def batch_predict(self, texts):
        """
        Predict multiple texts at once.

        Args:
            texts (list): List of news texts to classify

        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def main():
    """
    Main function to demonstrate the fake news detection system.
    """
    print("Fake News Detection using RoBERTa")
    print("=" * 50)

    # Initialize detector
    detector = FakeNewsDetector()

    # Load data (using sample data for demonstration)
    df = detector.load_data()

    # Prepare dataset
    train_dataset, val_dataset = detector.prepare_dataset(df)

    # Train model
    trainer = detector.train_model(train_dataset, val_dataset, num_epochs=2)

    # Evaluate model
    results = detector.evaluate_model(val_dataset, trainer)

    # Test predictions
    test_texts = [
        "Scientists have discovered that eating chocolate prevents all diseases",
        "The stock market closed with moderate gains today",
        "Aliens have invaded Earth and are demanding pizza",
        "Weather forecast shows rain expected tomorrow"
    ]

    print("\nTesting predictions:")
    print("-" * 30)

    predictions = detector.batch_predict(test_texts)
    for pred in predictions:
        print(f"Text: {pred['text'][:50]}...")
        print(f"Prediction: {pred['prediction']} (Confidence: {pred['confidence']:.3f})")
        print()

# Example usage with pre-trained model from HuggingFace
def use_pretrained_model():
    """
    Example of using a pre-trained model from HuggingFace Hub.
    """
    print("\nUsing Pre-trained Model from HuggingFace")
    print("=" * 50)

    # Initialize with pre-trained fake news detection model
    model_name = "hamzab/roberta-fake-news-classification"

    # Create pipeline
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

    # Test texts
    test_texts = [
        "<title>Scientists cure cancer<content>Amazing breakthrough in medical research<end>",
        "<title>Weather Update<content>Tomorrow will be sunny with temperature 75F<end>",
        "<title>Miracle Diet<content>Lose 50 pounds overnight with this one trick<end>"
    ]

    print("Testing with pre-trained model:")
    for text in test_texts:
        result = classifier(text)
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {result[0]['label']} (Score: {result[0]['score']:.3f})")
        print()

if __name__ == "__main__":
    main()
    use_pretrained_model()
