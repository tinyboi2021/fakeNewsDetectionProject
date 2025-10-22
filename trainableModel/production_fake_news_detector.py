#!/usr/bin/env python3
"""
Enhanced Production-Grade Fake News Detection System
- Focal Loss + Label Smoothing
- Attention Pooling + Multi-Sample Dropout  
- Layer-wise Learning Rates
- Cosine Scheduler with Warmup
- Gradient Accumulation
- Comprehensive Visualizations
- Auto-saves models and plots
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
import datetime
import re
import json
from typing import List, Dict

warnings.filterwarnings('ignore')

# ===================== ENHANCED CONFIGURATION =====================
CONFIG = {
    'model_name': 'roberta-base',
    'max_length': 384,  # Increased for news articles
    'batch_size': 8,
    'accumulation_steps': 4,  # Effective batch = 32
    'learning_rate': 1e-5,  # Lower for fine-tuning
    'num_epochs': 5,
    'n_folds': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 3,
    'random_seed': 42,
    'model_dir': './production_model',
    'results_dir': './results',
    'plots_dir': './plots',
    'checkpoints_dir': './checkpoints',
    'default_csv': '/app/src/dataset2/News_dataset/train_dataset.csv',
    
    # Advanced features
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'use_label_smoothing': True,
    'label_smoothing': 0.1,
    'use_attention_pooling': True,
    'use_multi_sample_dropout': True,
    'num_dropout_samples': 5,
    'dropout_rate': 0.2,
    'use_layer_wise_lr': True,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=CONFIG['default_csv'],
                   help="Path to CSV with 'content' and 'is_fakenews'")
    p.add_argument('--profile', choices=['low','high','production'], default='production',
                   help="Resource profile")
    p.add_argument('--no-interactive', action='store_true',
                   help="Skip interactive mode")
    return p.parse_known_args()[0]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return ' '.join(text.split())

# ===================== ENHANCED LOSSES =====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=inputs.size(-1)).float()
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / inputs.size(-1)
        
        if self.weight is not None:
            loss = -(targets_smooth * log_probs).sum(dim=-1) * self.weight[targets]
        else:
            loss = -(targets_smooth * log_probs).sum(dim=-1)
        return loss.mean()

# ===================== ENHANCED MODEL ARCHITECTURE =====================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states, attention_mask):
        attn_weights = self.attention(hidden_states).squeeze(-1)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)
        pooled = torch.bmm(attn_weights, hidden_states).squeeze(1)
        return pooled

class MultiSampleDropout(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rate, num_samples=5):
        super().__init__()
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size, num_labels) for _ in range(num_samples)])
    
    def forward(self, x):
        logits_list = []
        for dropout, classifier in zip(self.dropouts, self.classifiers):
            logits_list.append(classifier(dropout(x)))
        return torch.stack(logits_list, dim=0).mean(dim=0)

class EnhancedFakeNewsModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        if CONFIG['use_attention_pooling']:
            self.pooler = AttentionPooling(hidden_size)
        else:
            self.pooler = None
        
        if CONFIG['use_multi_sample_dropout']:
            self.classifier = MultiSampleDropout(
                hidden_size, num_labels, CONFIG['dropout_rate'], 
                CONFIG['num_dropout_samples']
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(CONFIG['dropout_rate']),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(CONFIG['dropout_rate']),
                nn.Linear(hidden_size, num_labels)
            )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        if self.pooler:
            pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        logits = self.classifier(pooled)
        return logits

# ===================== DATASET =====================
class FakeNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        enc = self.tokenizer.encode_plus(
            self.texts[i], add_special_tokens=True,
            max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[i], dtype=torch.long)
        }

# ===================== VISUALIZATION =====================
def plot_and_save_metrics(y_true, y_pred, y_prob, fold, epoch, split="validation"):
    """Save comprehensive visualizations"""
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Fold {fold} Epoch {epoch} - {split}\nConfusion Matrix')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    
    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title(f'Fold {fold} Epoch {epoch} - {split}\nROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    axes[2].plot(prob_pred, prob_true, marker='o', label='Model')
    axes[2].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axes[2].set_title(f'Fold {fold} Epoch {epoch} - {split}\nCalibration')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plots_dir']}/fold{fold}_epoch{epoch}_{split}_{timestamp}.png", dpi=150)
    plt.close()

# ===================== ENHANCED DETECTOR =====================
class ProductionFakeNewsDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        for d in ['model_dir', 'results_dir', 'plots_dir', 'checkpoints_dir']:
            os.makedirs(config[d], exist_ok=True)
        
        # Create loss function
        self.loss_fn = None
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        print("\n📂 Loading data...")
        df = pd.read_csv(csv_path)
        if 'content' not in df or 'is_fakenews' not in df:
            raise KeyError("CSV needs 'content' & 'is_fakenews'")
        
        df = df.rename(columns={'is_fakenews':'label'})
        df['processed_text'] = df['content'].apply(preprocess_text)
        df = df[df['processed_text'].str.len() > 20]
        df = df.drop_duplicates(subset=['processed_text'])
        
        print(f"✅ Loaded {len(df)} articles")
        print(f"   Class distribution: {df['label'].value_counts().to_dict()}")
        return df
    
    def get_optimizer_params(self, model):
        if not CONFIG['use_layer_wise_lr']:
            return model.parameters()
        
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        
        encoder_params = [
            {'params': [p for n, p in model.encoder.named_parameters() 
                       if not any(nd in n for nd in no_decay)],
             'lr': CONFIG['learning_rate'] * 0.1, 'weight_decay': CONFIG['weight_decay']},
            {'params': [p for n, p in model.encoder.named_parameters() 
                       if any(nd in n for nd in no_decay)],
             'lr': CONFIG['learning_rate'] * 0.1, 'weight_decay': 0.0}
        ]
        
        classifier_params = [
            {'params': [p for n, p in model.named_parameters() 
                       if 'classifier' in n and not any(nd in n for nd in no_decay)],
             'lr': CONFIG['learning_rate'], 'weight_decay': CONFIG['weight_decay']},
            {'params': [p for n, p in model.named_parameters() 
                       if 'classifier' in n and any(nd in n for nd in no_decay)],
             'lr': CONFIG['learning_rate'], 'weight_decay': 0.0}
        ]
        
        return encoder_params + classifier_params
    
    def train_fold(self, train_loader, val_loader, class_weights, fold):
        print(f"\n{'='*60}")
        print(f"📊 Training Fold {fold+1}/{self.config['n_folds']}")
        print(f"{'='*60}")
        
        model = EnhancedFakeNewsModel(self.config['model_name']).to(self.device)
        
        # Setup loss
        if CONFIG['use_focal_loss']:
            self.loss_fn = FocalLoss(CONFIG['focal_alpha'], CONFIG['focal_gamma'], class_weights)
        elif CONFIG['use_label_smoothing']:
            self.loss_fn = LabelSmoothingCrossEntropy(CONFIG['label_smoothing'], class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = AdamW(self.get_optimizer_params(model))
        
        total_steps = len(train_loader) * self.config['num_epochs'] // CONFIG['accumulation_steps']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        best_loss, patience = 1e9, 0
        best_state = None
        
        for epoch in range(self.config['num_epochs']):
            # Training
            model.train()
            tr_loss = 0
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels) / CONFIG['accumulation_steps']
                loss.backward()
                
                if (step + 1) % CONFIG['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                tr_loss += loss.item() * CONFIG['accumulation_steps']
                pbar.set_postfix({'loss': f'{tr_loss/(step+1):.4f}'})
            
            # Validation
            val_metrics = self.evaluate(model, val_loader)
            avg_tr = tr_loss / len(train_loader)
            
            print(f"\n📈 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {avg_tr:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
            
            # Plot
            plot_and_save_metrics(val_metrics['labels'], val_metrics['preds'], 
                                val_metrics['probs'], fold+1, epoch+1)
            
            # Save best
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_state = model.state_dict()
                patience = 0
                torch.save(best_state, f"{self.config['checkpoints_dir']}/fold{fold}.pt")
                print(f"   ✅ Best model saved")
            else:
                patience += 1
                if patience >= self.config['early_stopping_patience']:
                    print(f"   ⏸️  Early stopping")
                    break
        
        model.load_state_dict(best_state)
        final_metrics = self.evaluate(model, val_loader)
        return model, final_metrics
    
    def evaluate(self, model, loader):
        model.eval()
        losses, preds, labs, probs = [], [], [], []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)
                
                losses.append(loss.item())
                prob = torch.softmax(logits, 1)
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                labs.extend(labels.cpu().numpy())
                probs.extend(prob[:, 1].cpu().numpy())
        
        return {
            'loss': np.mean(losses),
            'accuracy': accuracy_score(labs, preds),
            'f1_score': f1_score(labs, preds, average='weighted'),
            'roc_auc': roc_auc_score(labs, probs),
            'brier_score': brier_score_loss(labs, probs),
            'preds': preds,
            'labels': labs,
            'probs': probs
        }
    
    def train_kfold(self, df):
        set_seed(CONFIG['random_seed'])
        
        X, y = df['processed_text'].values, df['label'].values
        class_weights = torch.tensor(
            compute_class_weight('balanced', classes=np.unique(y), y=y),
            dtype=torch.float
        ).to(self.device)
        
        skf = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True, 
                            random_state=CONFIG['random_seed'])
        
        models, metrics = [], []
        
        for i, (tr, va) in enumerate(skf.split(X, y)):
            tr_ds = FakeNewsDataset(list(X[tr]), list(y[tr]), self.tokenizer, CONFIG['max_length'])
            va_ds = FakeNewsDataset(list(X[va]), list(y[va]), self.tokenizer, CONFIG['max_length'])
            
            tr_ld = DataLoader(tr_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
            va_ld = DataLoader(va_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
            
            m, mt = self.train_fold(tr_ld, va_ld, class_weights, i)
            models.append(m)
            metrics.append(mt)
        
        self.models, self.fold_metrics = models, metrics
        
        # Summary
        avg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0] if k not in ['preds', 'labels', 'probs']}
        std = {k: np.std([m[k] for m in metrics]) for k in metrics[0] if k not in ['preds', 'labels', 'probs']}
        
        print(f"\n{'='*60}")
        print(f"📊 CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        for k in avg:
            print(f"{k:15s}: {avg[k]:.4f} ± {std[k]:.4f}")
        print(f"{'='*60}")
        
        return avg
    
    def save_best_model(self):
        best_idx = np.argmax([m['accuracy'] for m in self.fold_metrics])
        best_model = self.models[best_idx]
        
        # Save encoder and tokenizer
        best_model.encoder.save_pretrained(f"{CONFIG['model_dir']}/best_model")
        self.tokenizer.save_pretrained(f"{CONFIG['model_dir']}/best_model")
        
        # Save full state for loading
        torch.save({
            'model_state': best_model.state_dict(),
            'config': CONFIG,
            'metrics': self.fold_metrics[best_idx]
        }, f"{CONFIG['model_dir']}/best_model/full_model.pt")
        
        print(f"\n💾 Best model (Fold {best_idx+1}) saved to {CONFIG['model_dir']}/best_model")
        print(f"   Accuracy: {self.fold_metrics[best_idx]['accuracy']:.4f}")
    
    def load_best_model(self):
        checkpoint = torch.load(f"{CONFIG['model_dir']}/best_model/full_model.pt")
        model = EnhancedFakeNewsModel(CONFIG['model_name']).to(self.device)
        model.load_state_dict(checkpoint['model_state'])
        return model, self.tokenizer
    
    def predict_single(self, text, model=None, tokenizer=None):
        if model is None:
            model, tokenizer = self.load_best_model()
        
        model.eval()
        txt = preprocess_text(text)
        enc = tokenizer.encode_plus(txt, add_special_tokens=True,
                                    max_length=CONFIG['max_length'],
                                    padding='max_length', truncation=True,
                                    return_tensors='pt')
        
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, 1)
            pred = torch.argmax(logits, 1).item()
        
        return {
            'text': text,
            'prediction': 'Real' if pred == 1 else 'Fake',
            'confidence': probs[0][pred].item(),
            'fake_probability': probs[0][0].item(),
            'real_probability': probs[0][1].item()
        }
    
    def interactive_mode(self):
        model, tok = self.load_best_model()
        st = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sf = f"{CONFIG['results_dir']}/session_{st}.txt"
        
        with open(sf, 'w') as f:
            f.write(f"Session {st}\n")
        
        cnt = 0
        print("\n" + "="*60)
        print("🎯 Interactive Fake News Detection")
        print("="*60)
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                txt = input("📝 Enter news text: ").strip()
                if txt.lower() == 'exit':
                    break
                
                if not txt:
                    continue
                
                res = self.predict_single(txt, model, tok)
                cnt += 1
                
                print(f"\n🔍 Result:")
                print(f"   Prediction: {res['prediction']}")
                print(f"   Confidence: {res['confidence']:.2%}")
                print(f"   Fake Prob: {res['fake_probability']:.2%}")
                print(f"   Real Prob: {res['real_probability']:.2%}\n")
                
                with open(sf, 'a') as f:
                    f.write(json.dumps(res) + "\n")
            
            except KeyboardInterrupt:
                break
        
        print(f"\n💾 Logged {cnt} predictions to {sf}")

def main():
    print("="*60)
    print("🚀 ENHANCED PRODUCTION FAKE NEWS DETECTOR")
    print("="*60)
    
    args = parse_args()
    
    # Apply profile
    if args.profile == 'low':
        CONFIG.update(batch_size=8, max_length=256, num_epochs=3)
    elif args.profile == 'high':
        CONFIG.update(batch_size=16, max_length=512, num_epochs=7, n_folds=10)
    
    set_seed(CONFIG['random_seed'])
    detector = ProductionFakeNewsDetector(CONFIG)
    
    df = detector.load_data(args.csv)
    detector.train_kfold(df)
    detector.save_best_model()
    
    if not args.no_interactive:
        try:
            choice = input("\n🎮 Enter interactive mode? (y/n): ").strip().lower()
            if choice == 'y':
                detector.interactive_mode()
        except:
            pass
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
