#!/usr/bin/env python3
"""
Production-Grade Fake News Detection System
Combines robust K-fold training with interactive user interface
Uses RoBERTa with advanced training techniques and comprehensive evaluation
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    roc_auc_score, brier_score_loss, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import re
from typing import Tuple, List, Dict
import json

warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
CONFIG = {
    'model_name': 'roberta-base',
    'max_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 4,
    'n_folds': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 2,
    'random_seed': 42,
    'model_dir': './production_model',
    'results_dir': './results',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ===================== UTILITY FUNCTIONS =====================
def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

# ===================== DATASET CLASS =====================
class FakeNewsDataset(Dataset):
    """Custom dataset for fake news detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ===================== TRAINER CLASS =====================
class ProductionFakeNewsDetector:
    """Production-grade fake news detector with K-fold validation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.models = []
        self.fold_metrics = []
        
        # Create directories
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['results_dir'], exist_ok=True)
        
        print(f"🚀 Initialized Production Fake News Detector")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {config['model_name']}")
    
    def load_data(self, fake_csv_path: str, true_csv_path: str) -> pd.DataFrame:
        """Load and prepare dataset"""
        print("\n📂 Loading data...")
        fake_df = pd.read_csv(fake_csv_path)
        true_df = pd.read_csv(true_csv_path)
        
        fake_df['label'] = 0
        true_df['label'] = 1
        
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Combine title and text
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['processed_text'] = df['combined_text'].apply(preprocess_text)
        
        # Remove short texts
        df = df[df['processed_text'].str.len() > 10]
        
        print(f"✅ Loaded {len(df)} articles")
        print(f"   - Real news: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
        print(f"   - Fake news: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
        
        return df
    
    def train_fold(self, train_loader: DataLoader, val_loader: DataLoader, 
                   class_weights: torch.Tensor, fold: int) -> Dict:
        """Train a single fold"""
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=2
        ).to(self.device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config['num_epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print(f"Training Fold {fold + 1}/{self.config['n_folds']}")
        print(f"{'='*60}")
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = criterion(outputs.logits, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config['max_grad_norm']
                )
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                          f"Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f}")
            
            # Calculate training metrics
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            
            # Validation phase
            val_loss, val_acc, val_f1, val_auc = self.evaluate(model, val_loader, criterion)
            
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model for this fold
                torch.save(model.state_dict(), 
                          f"{self.config['model_dir']}/fold_{fold}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"⚠️  Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f"{self.config['model_dir']}/fold_{fold}_best.pt"))
        
        # Final evaluation
        final_metrics = self.comprehensive_evaluate(model, val_loader)
        
        return model, final_metrics
    
    def evaluate(self, model, data_loader, criterion):
        """Evaluate model on validation set"""
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        val_loss /= len(data_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.0
        
        return val_loss, val_acc, val_f1, val_auc
    
    def comprehensive_evaluate(self, model, data_loader):
        """Comprehensive evaluation with multiple metrics"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, average='weighted'),
            'roc_auc': roc_auc_score(all_labels, all_probs),
            'brier_score': brier_score_loss(all_labels, all_probs),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return metrics
    
    def train_kfold(self, df: pd.DataFrame):
        """Train with K-fold cross-validation"""
        set_seed(self.config['random_seed'])
        
        X = df['processed_text'].values
        y = df['label'].values
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        print(f"\n⚖️  Class weights: Fake={class_weights[0]:.2f}, Real={class_weights[1]:.2f}")
        
        # K-fold split
        skf = StratifiedKFold(
            n_splits=self.config['n_folds'],
            shuffle=True,
            random_state=self.config['random_seed']
        )
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_texts = X[train_idx].tolist()
            train_labels = y[train_idx].tolist()
            val_texts = X[val_idx].tolist()
            val_labels = y[val_idx].tolist()
            
            # Create datasets
            train_dataset = FakeNewsDataset(
                train_texts, train_labels,
                self.tokenizer, self.config['max_length']
            )
            val_dataset = FakeNewsDataset(
                val_texts, val_labels,
                self.tokenizer, self.config['max_length']
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )
            
            # Train fold
            model, metrics = self.train_fold(
                train_loader, val_loader, class_weights, fold
            )
            
            self.models.append(model)
            self.fold_metrics.append(metrics)
            
            print(f"\n✅ Fold {fold+1} Results:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   Brier Score: {metrics['brier_score']:.4f}")
        
        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in self.fold_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in self.fold_metrics]),
            'roc_auc': np.mean([m['roc_auc'] for m in self.fold_metrics]),
            'brier_score': np.mean([m['brier_score'] for m in self.fold_metrics])
        }
        
        print(f"\n{'='*60}")
        print(f"📊 CROSS-VALIDATION RESULTS (Average across {self.config['n_folds']} folds)")
        print(f"{'='*60}")
        print(f"Accuracy:    {avg_metrics['accuracy']:.4f} ± {np.std([m['accuracy'] for m in self.fold_metrics]):.4f}")
        print(f"F1-Score:    {avg_metrics['f1_score']:.4f} ± {np.std([m['f1_score'] for m in self.fold_metrics]):.4f}")
        print(f"ROC-AUC:     {avg_metrics['roc_auc']:.4f} ± {np.std([m['roc_auc'] for m in self.fold_metrics]):.4f}")
        print(f"Brier Score: {avg_metrics['brier_score']:.4f} ± {np.std([m['brier_score'] for m in self.fold_metrics]):.4f}")
        print(f"{'='*60}")
        
        # Save metrics
        with open(f"{self.config['results_dir']}/kfold_metrics.json", 'w') as f:
            json.dump({
                'fold_metrics': [
                    {k: float(v) if isinstance(v, (np.floating, float)) else v 
                     for k, v in m.items() if k not in ['predictions', 'labels', 'probabilities']}
                    for m in self.fold_metrics
                ],
                'average_metrics': avg_metrics
            }, f, indent=2)
        
        return avg_metrics
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix (using first fold)
        cm = confusion_matrix(
            self.fold_metrics[0]['labels'],
            self.fold_metrics[0]['predictions']
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        axes[0, 0].set_title('Confusion Matrix (Fold 1)')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve (all folds)
        for i, metrics in enumerate(self.fold_metrics):
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
            axes[0, 1].plot(fpr, tpr, label=f'Fold {i+1} (AUC = {metrics["roc_auc"]:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves - All Folds')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Metrics Comparison Across Folds
        metrics_names = ['Accuracy', 'F1-Score', 'ROC-AUC']
        metrics_values = [
            [m['accuracy'] for m in self.fold_metrics],
            [m['f1_score'] for m in self.fold_metrics],
            [m['roc_auc'] for m in self.fold_metrics]
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.15
        for i, fold_values in enumerate(zip(*metrics_values)):
            axes[1, 0].bar(x + i*width, fold_values, width, label=f'Fold {i+1}')
        
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Metrics Comparison Across Folds')
        axes[1, 0].set_xticks(x + width * 2)
        axes[1, 0].set_xticklabels(metrics_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Calibration Curve (first fold)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.fold_metrics[0]['labels'],
            self.fold_metrics[0]['probabilities'],
            n_bins=10
        )
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['results_dir']}/comprehensive_results.png", dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualizations saved to {self.config['results_dir']}/comprehensive_results.png")
        plt.close()
    
    def save_best_model(self):
        """Save the best performing fold model"""
        best_fold = np.argmax([m['accuracy'] for m in self.fold_metrics])
        best_model = self.models[best_fold]
        
        # Save model and tokenizer
        best_model.save_pretrained(f"{self.config['model_dir']}/best_model")
        self.tokenizer.save_pretrained(f"{self.config['model_dir']}/best_model")
        
        print(f"\n💾 Best model (Fold {best_fold+1}) saved to {self.config['model_dir']}/best_model")
        print(f"   Accuracy: {self.fold_metrics[best_fold]['accuracy']:.4f}")
    
    def load_best_model(self):
        """Load the saved best model"""
        model_path = f"{self.config['model_dir']}/best_model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model found at {model_path}")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"✅ Loaded best model from {model_path}")
        return model, tokenizer
    
    def predict_single(self, text: str, model=None, tokenizer=None) -> Dict:
        """Predict on a single text with detailed output"""
        if model is None or tokenizer is None:
            model, tokenizer = self.load_best_model()
        
        model.eval()
        
        # Preprocess
        processed_text = preprocess_text(text)
        
        # Tokenize
        encoding = tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(outputs.logits, dim=1)
        
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()
        prediction = "Fake" if pred.item() == 0 else "Real"
        confidence = max(fake_prob, real_prob)
        
        return {
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob
        }
    
    def interactive_mode(self):
        """Interactive prediction interface"""
        print("\n" + "="*60)
        print("🎯 INTERACTIVE FAKE NEWS DETECTION")
        print("="*60)
        print("Enter news text to analyze. Type 'exit' to quit.")
        print("-" * 60)
        
        # Load best model
        model, tokenizer = self.load_best_model()
        
        # Create session file
        session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = f"{self.config['results_dir']}/prediction_session_{session_timestamp}.txt"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            f.write("FAKE NEWS DETECTION SESSION\n")
            f.write(f"Session started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
        
        prediction_count = 0
        
        while True:
            try:
                user_input = input("\n📝 Enter news text (or 'exit'): ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n👋 Exiting interactive mode.")
                    break
                
                if not user_input:
                    print("⚠️  Please enter some text.")
                    continue
                
                print("🤔 Analyzing...")
                prediction = self.predict_single(user_input, model, tokenizer)
                prediction_count += 1
                
                # Print results
                print("\n" + "="*60)
                print("🔍 PREDICTION RESULTS")
                print("="*60)
                print(f"📝 Input: {prediction['text'][:100]}...")
                print(f"🎯 Prediction: {prediction['prediction']}")
                print(f"🔢 Confidence: {prediction['confidence']:.4f} ({prediction['confidence']*100:.2f}%)")
                print(f"🚨 Fake Probability: {prediction['fake_probability']:.4f}")
                print(f"✅ Real Probability: {prediction['real_probability']:.4f}")
                
                # Confidence level
                if prediction['confidence'] >= 0.9:
                    confidence_level = "🟢 Very High"
                elif prediction['confidence'] >= 0.7:
                    confidence_level = "🟡 High"
                elif prediction['confidence'] >= 0.5:
                    confidence_level = "🟠 Moderate"
                else:
                    confidence_level = "🔴 Low"
                
                print(f"📊 Confidence Level: {confidence_level}")
                print("="*60)
                
                # Save to file
                with open(session_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Prediction #{prediction_count}\n")
                    f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Input: {prediction['text']}\n")
                    f.write(f"Prediction: {prediction['prediction']}\n")
                    f.write(f"Confidence: {prediction['confidence']:.4f}\n")
                    f.write(f"Fake Probability: {prediction['fake_probability']:.4f}\n")
                    f.write(f"Real Probability: {prediction['real_probability']:.4f}\n")
                
                print(f"💾 Results saved to: {session_file}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
        
        # Session summary
        with open(session_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Session ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total predictions: {prediction_count}\n")
            f.write("="*60 + "\n")
        
        print(f"\n📁 Session log: {session_file}")
        print(f"📊 Total predictions: {prediction_count}")

# ===================== MAIN FUNCTION =====================
def main():
    """Main execution function"""
    print("="*60)
    print("🚀 PRODUCTION-GRADE FAKE NEWS DETECTION SYSTEM")
    print("="*60)
    print("Combining K-fold validation with interactive interface")
    print("="*60)
    
    # Initialize detector
    detector = ProductionFakeNewsDetector(CONFIG)
    
    # Check if model exists
    model_exists = os.path.exists(f"{CONFIG['model_dir']}/best_model")
    
    if not model_exists:
        print("\n📚 No pre-trained model found. Starting training...")
        
        # Load data
        df = detector.load_data(
            fake_csv_path="/app/src/dataset1/News_dataset/Fake.csv",
            true_csv_path="/app/src/dataset1/News_dataset/True.csv"
        )
        
        # Train with K-fold
        avg_metrics = detector.train_kfold(df)
        
        # Create visualizations
        detector.visualize_results()
        
        # Save best model
        detector.save_best_model()
        
        print("\n✅ Training completed successfully!")
    else:
        print("\n✅ Pre-trained model found. Skipping training.")
    
    # Interactive mode
    print("\n" + "="*60)
    choice = input("Enter interactive mode? (y/n): ").strip().lower()
    if choice == 'y':
        detector.interactive_mode()
    else:
        print("👋 Exiting without interactive mode.")

if __name__ == "__main__":
    set_seed(CONFIG['random_seed'])
    main()