#!/usr/bin/env python3
"""
Enhanced Clickbait Detector with Transfer Learning from Fake News Model
- Loads pretrained fake news detector as base
- Fine-tunes on clickbait data
- Uses all advanced techniques from production model
- Saves models and visualizations
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
import re
import json
import datetime
from typing import List, Dict

warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
CONFIG = {
    'source_model_path': './production_model/best_model',  # Fake news model
    'max_length': 384,
    'batch_size': 8,
    'accumulation_steps': 4,
    'learning_rate': 5e-6,  # Even lower for transfer learning
    'num_epochs': 5,
    'n_folds': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 3,
    'random_seed': 42,
    'model_dir': './clickbait_model',
    'results_dir': './results_clickbait',
    'plots_dir': './plots_clickbait',
    'checkpoints_dir': './checkpoints_clickbait',
    
    # Advanced features (same as fake news model)
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
    p.add_argument('--csv', required=True, help="Path to clickbait CSV")
    p.add_argument('--source-model', default=CONFIG['source_model_path'],
                   help="Path to pretrained fake news model")
    p.add_argument('--test-csv', help="Path to test CSV")
    p.add_argument('--profile', choices=['low','high','production'], default='production')
    p.add_argument('--no-interactive', action='store_true')
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
    text = str(text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text[:1000]  # Limit length

# ===================== LOSSES (same as production) =====================
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

# ===================== MODEL (same as production) =====================
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

class EnhancedClickbaitModel(nn.Module):
    def __init__(self, source_model_path, num_labels=2):
        super().__init__()
        
        # Load encoder from pretrained fake news model
        print(f"🔄 Loading pretrained encoder from: {source_model_path}")
        self.encoder = AutoModel.from_pretrained(source_model_path)
        hidden_size = self.encoder.config.hidden_size
        
        # New classification head for clickbait
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
        
        print("✅ Transfer learning model initialized")
        print(f"   Encoder: Pretrained from fake news detector")
        print(f"   Classifier: New for clickbait detection")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        if self.pooler:
            pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled)
        return logits

# ===================== DATASET =====================
class ClickbaitDataset(Dataset):
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
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0])
    axes[0].set_title(f'Clickbait - Fold {fold} Epoch {epoch}\n{split} Confusion Matrix')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    
    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='darkorange')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title(f'Clickbait - Fold {fold} Epoch {epoch}\n{split} ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    axes[2].plot(prob_pred, prob_true, marker='o', label='Model', color='darkorange')
    axes[2].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axes[2].set_title(f'Clickbait - Fold {fold} Epoch {epoch}\n{split} Calibration')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plots_dir']}/fold{fold}_epoch{epoch}_{split}_{timestamp}.png", dpi=150)
    plt.close()

# ===================== TRAINER =====================
class ClickbaitTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"🎯 CLICKBAIT DETECTOR WITH TRANSFER LEARNING")
        print(f"{'='*60}")
        print(f"🚀 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load tokenizer from source model
        self.tokenizer = AutoTokenizer.from_pretrained(config['source_model_path'])
        
        for d in ['model_dir', 'results_dir', 'plots_dir', 'checkpoints_dir']:
            os.makedirs(config[d], exist_ok=True)
        
        self.loss_fn = None
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        print(f"\n📂 Loading clickbait data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Handle different column names
        if 'content' in df.columns:
            text_col = 'content'
        elif 'title' in df.columns:
            text_col = 'title'
        elif 'text' in df.columns:
            text_col = 'text'
        else:
            raise KeyError("Need 'content', 'title', or 'text' column")
        
        if 'is_clickbait' in df.columns:
            label_col = 'is_clickbait'
        elif 'is_fakenews' in df.columns:
            label_col = 'is_fakenews'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            raise KeyError("Need 'is_clickbait', 'is_fakenews', or 'label' column")
        
        df = df.rename(columns={text_col: 'text', label_col: 'label'})
        df['processed_text'] = df['text'].apply(preprocess_text)
        df = df[df['processed_text'].str.len() > 10]
        df = df.drop_duplicates(subset=['processed_text'])
        
        print(f"✅ Loaded {len(df)} clickbait samples")
        print(f"   Class distribution: {df['label'].value_counts().to_dict()}")
        return df
    
    def get_optimizer_params(self, model):
        if not CONFIG['use_layer_wise_lr']:
            return model.parameters()
        
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        
        # Very low LR for pretrained encoder
        encoder_params = [
            {'params': [p for n, p in model.encoder.named_parameters() 
                       if not any(nd in n for nd in no_decay)],
             'lr': CONFIG['learning_rate'] * 0.1, 'weight_decay': CONFIG['weight_decay']},
            {'params': [p for n, p in model.encoder.named_parameters() 
                       if any(nd in n for nd in no_decay)],
             'lr': CONFIG['learning_rate'] * 0.1, 'weight_decay': 0.0}
        ]
        
        # Normal LR for new classifier
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
        
        model = EnhancedClickbaitModel(self.config['source_model_path']).to(self.device)
        
        # Setup loss
        if CONFIG['use_focal_loss']:
            self.loss_fn = FocalLoss(CONFIG['focal_alpha'], CONFIG['focal_gamma'], class_weights)
            print("📊 Using Focal Loss")
        elif CONFIG['use_label_smoothing']:
            self.loss_fn = LabelSmoothingCrossEntropy(CONFIG['label_smoothing'], class_weights)
            print("📊 Using Label Smoothing")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("📊 Using Cross Entropy")
        
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
            tr_ds = ClickbaitDataset(list(X[tr]), list(y[tr]), self.tokenizer, CONFIG['max_length'])
            va_ds = ClickbaitDataset(list(X[va]), list(y[va]), self.tokenizer, CONFIG['max_length'])
            
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
        print(f"📊 CLICKBAIT CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        for k in avg:
            print(f"{k:15s}: {avg[k]:.4f} ± {std[k]:.4f}")
        print(f"{'='*60}")
        
        return avg
    
    def save_best_model(self):
        best_idx = np.argmax([m['accuracy'] for m in self.fold_metrics])
        best_model = self.models[best_idx]
        
        best_model.encoder.save_pretrained(f"{CONFIG['model_dir']}/best_model")
        self.tokenizer.save_pretrained(f"{CONFIG['model_dir']}/best_model")
        
        torch.save({
            'model_state': best_model.state_dict(),
            'config': CONFIG,
            'metrics': self.fold_metrics[best_idx]
        }, f"{CONFIG['model_dir']}/best_model/full_model.pt")
        
        print(f"\n💾 Best clickbait model (Fold {best_idx+1}) saved")
        print(f"   Accuracy: {self.fold_metrics[best_idx]['accuracy']:.4f}")
    
    def test_on_external(self, test_csv):
        print(f"\n{'='*60}")
        print(f"🧪 Testing on External Test Set")
        print(f"{'='*60}")
        
        test_df = self.load_data(test_csv)
        test_ds = ClickbaitDataset(
            test_df['processed_text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            CONFIG['max_length']
        )
        test_ld = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        best_idx = np.argmax([m['accuracy'] for m in self.fold_metrics])
        model = self.models[best_idx]
        
        test_metrics = self.evaluate(model, test_ld)
        
        print(f"\n📈 Test Set Performance:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:  {test_metrics['roc_auc']:.4f}")
        print(f"   Brier:    {test_metrics['brier_score']:.4f}")
        print(f"\n{classification_report(test_metrics['labels'], test_metrics['preds'], digits=4)}")
        
        plot_and_save_metrics(test_metrics['labels'], test_metrics['preds'], 
                            test_metrics['probs'], 0, 0, "test")

def main():
    args = parse_args()
    
    # Apply profile
    if args.profile == 'low':
        CONFIG.update(batch_size=8, max_length=256, num_epochs=3)
    elif args.profile == 'high':
        CONFIG.update(batch_size=16, max_length=512, num_epochs=7, n_folds=10)
    
    CONFIG['source_model_path'] = args.source_model
    
    trainer = ClickbaitTrainer(CONFIG)
    df = trainer.load_data(args.csv)
    trainer.train_kfold(df)
    trainer.save_best_model()
    
    if args.test_csv:
        trainer.test_on_external(args.test_csv)
    
    print("\n✅ Clickbait training complete!")

if __name__ == "__main__":
    main()
