import os
import torch
import pandas as pd
import numpy as np
import re
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
import warnings
import datetime
from datasets import Dataset

warnings.filterwarnings('ignore')

class FakeNewsDetector:
    def __init__(self, model_name="roberta-base", model_dir="./fake_news_roberta"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def load_data(self, fake_csv_path="/app/src/dataset1/News_dataset/Fake.csv", true_csv_path="/app/src/dataset1/News_dataset/True.csv"):
        if os.path.exists(self.model_dir):
            print(f"Loading data from disk skipped as model directory exists at {self.model_dir}")
            return None  # Skip loading data on retrain

        fake_df = pd.read_csv(fake_csv_path)
        true_df = pd.read_csv(true_csv_path)
        fake_df['label'] = 0
        true_df['label'] = 1
        df = pd.concat([fake_df, true_df], ignore_index=True)
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        df = df[df['processed_text'].str.len() > 10]
        print(f"Dataset loaded: {len(df)} articles")
        print(f"Real news: {sum(df['label'] == 1)}")
        print(f"Fake news: {sum(df['label'] == 0)}")
        return df

    def prepare_dataset(self, df, test_size=0.2, max_length=512):
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['processed_text'].tolist(), df['label'].tolist(),
            test_size=test_size, random_state=42, stratify=df['label']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_encodings = self.tokenizer(
            train_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt'
        )
        val_encodings = self.tokenizer(
            val_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt'
        )
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

    def train_model(self, train_dataset, val_dataset, output_dir=None, num_epochs=3, learning_rate=2e-5, batch_size=16):
        output_dir = output_dir or self.model_dir

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, output_attentions=False, output_hidden_states=False
        )
        self.model.to(self.device)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            seed=42,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        return trainer

    def evaluate_model(self, val_dataset, trainer=None):
        if trainer:
            results = trainer.evaluate()
            print("\nEvaluation Results:")
            for key, value in results.items():
                print(f"{key}: {value:.4f}")

            predictions = trainer.predict(val_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
        else:
            if not self.model:
                raise ValueError("No model loaded. Train a model first or load a pre-trained one.")
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

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'])
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
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

    def load_trained_model(self, model_path=None):
        model_path = model_path or self.model_dir
        if not os.path.exists(model_path):
            print(f"No pre-trained model found at {model_path}")
            return False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")
        return True

    def predict(self, text):
        if not self.model or not self.tokenizer:
            raise ValueError("No model loaded. Train a model first or load a pre-trained one.")
        processed_text = self.preprocess_text(text)
        inputs = self.tokenizer(processed_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
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
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def print_prediction_details(prediction):
    """Print detailed prediction results to terminal"""
    print("\n" + "="*60)
    print("🔍 PREDICTION RESULTS")
    print("="*60)
    print(f"📝 Input Text: {prediction['text']}")
    print(f"🎯 Prediction: {prediction['prediction']}")
    print(f"🔢 Confidence: {prediction['confidence']:.4f} ({prediction['confidence']*100:.2f}%)")
    print(f"🚨 Fake Probability: {prediction['fake_probability']:.4f} ({prediction['fake_probability']*100:.2f}%)")
    print(f"✅ Real Probability: {prediction['real_probability']:.4f} ({prediction['real_probability']*100:.2f}%)")
    
    # Visual confidence indicator
    confidence_level = ""
    if prediction['confidence'] >= 0.9:
        confidence_level = "🟢 Very High"
    elif prediction['confidence'] >= 0.7:
        confidence_level = "🟡 High"
    elif prediction['confidence'] >= 0.5:
        confidence_level = "🟠 Moderate"
    else:
        confidence_level = "🔴 Low"
    
    print(f"📊 Confidence Level: {confidence_level}")
    
    # Add interpretation
    if prediction['prediction'] == 'Fake':
        print("⚠️  INTERPRETATION: This content appears to be FAKE NEWS")
        print("   Recommendation: Verify with reliable sources before sharing")
    else:
        print("✅ INTERPRETATION: This content appears to be LEGITIMATE")
        print("   Recommendation: Content seems credible, but always cross-check important information")
    
    print("="*60)

def save_prediction_to_file(prediction, output_file):
    """Save prediction results to file with enhanced formatting"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n")
        f.write(f"Input: {prediction['text']}\n")
        f.write(f"Prediction: {prediction['prediction']}\n")
        f.write(f"Confidence: {prediction['confidence']:.4f} ({prediction['confidence']*100:.2f}%)\n")
        f.write(f"Fake Probability: {prediction['fake_probability']:.4f} ({prediction['fake_probability']*100:.2f}%)\n")
        f.write(f"Real Probability: {prediction['real_probability']:.4f} ({prediction['real_probability']*100:.2f}%)\n")
        
        if prediction['prediction'] == 'Fake':
            f.write("Interpretation: FAKE NEWS - Verify with reliable sources\n")
        else:
            f.write("Interpretation: LEGITIMATE - Content seems credible\n")
        
        f.write("\n" + "="*60 + "\n\n")

def main():
    print("🔍 Fake News Detection using RoBERTa")
    print("=" * 50)

    detector = FakeNewsDetector()

    # Check if saved model exists and load to skip training
    if detector.load_trained_model():
        print("✅ Using pre-trained model, skipping training.")
    else:
        print("🚀 No pre-trained model found, training now...")
        df = detector.load_data()
        if df is not None:
            train_ds, val_ds = detector.prepare_dataset(df)
            trainer = detector.train_model(train_ds, val_ds, num_epochs=2)
            detector.evaluate_model(val_ds, trainer)

    print("\n🎯 Interactive Fake News Detection")
    print("Enter news text to analyze. Type 'exit' to quit.")
    print("-" * 50)

    # Create a single output file for the session
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_output_file = f"prediction_session_{session_timestamp}.txt"
    
    # Write session header
    with open(session_output_file, 'w', encoding='utf-8') as f:
        f.write("FAKE NEWS DETECTION SESSION\n")
        f.write(f"Session started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

    prediction_count = 0

    # User input for fake news detection
    while True:
        try:
            user_input = input("\n📝 Enter a news text to classify (or type 'exit' to quit): ").strip()
            if user_input.lower() == 'exit':
                print("\n👋 Exiting prediction loop.")
                break
            
            if not user_input:
                print("⚠️  Please enter some text to analyze.")
                continue

            print("🤔 Analyzing...")
            prediction = detector.predict(user_input)
            prediction_count += 1
            
            # Print detailed results to terminal
            print_prediction_details(prediction)
            
            # Save to session file
            save_prediction_to_file(prediction, session_output_file)
            
            print(f"💾 Results saved to: {session_output_file}")
            print(f"📊 Total predictions this session: {prediction_count}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error occurred: {str(e)}")
            continue

    # Write session footer
    with open(session_output_file, 'a', encoding='utf-8') as f:
        f.write(f"\nSession ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total predictions: {prediction_count}\n")
        f.write("="*60 + "\n")

    print(f"\n📁 Complete session log saved to: {session_output_file}")
    print("✨ Thank you for using Fake News Detection!")

if __name__ == "__main__":
    main()