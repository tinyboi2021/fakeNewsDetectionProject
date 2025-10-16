#!/usr/bin/env python3
"""
Production-Grade Fake News Detection System
Combines robust K-fold training with interactive user interface
Uses RoBERTa with advanced training techniques and comprehensive evaluation
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
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import re
import json
from typing import List, Dict

warnings.filterwarnings('ignore')

# ===================== CONFIGURATION =====================
CONFIG = {
    'model_name': 'roberta-base',
    'max_length': 128,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'n_folds': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 2,
    'random_seed': 42,
    'model_dir': './production_model',
    'results_dir': './results',
    'default_csv': '/app/src/dataset2/News_dataset/train_dataset.csv'
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=CONFIG['default_csv'],
                   help="Path to CSV with 'content' and 'is_fakenews'")
    p.add_argument('--profile', choices=['low','high'], default='low',
                   help="Resource profile: low=4060Ti, high=A100")
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return ' '.join(text.split())

class FakeNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts, self.labels, self.tokenizer, self.max_length = texts, labels, tokenizer, max_length
    def __len__(self): return len(self.texts)
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

class ProductionFakeNewsDetector:
    def __init__(self, config: Dict):
        self.config, self.device = config, torch.device('cpu')
        # GPU selection
        gpus = torch.cuda.device_count()
        if gpus:
            choice = os.environ.get('CUDA_VISIBLE_DEVICES','')
            self.device = torch.device('cuda:'+choice if choice else 'cuda')
        print(f"🚀 Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['results_dir'], exist_ok=True)
    def load_data(self, csv_path: str) -> pd.DataFrame:
        print("\n📂 Loading data...")
        df = pd.read_csv(csv_path)
        if 'content' not in df or 'is_fakenews' not in df:
            raise KeyError("CSV needs 'content' & 'is_fakenews'")
        df = df.rename(columns={'is_fakenews':'label'})
        df['processed_text'] = df['content'].apply(preprocess_text)
        df = df[df['processed_text'].str.len()>10]
        print(f"✅ Loaded {len(df)} articles")
        return df
    def train_fold(self, train_loader, val_loader, cw, fold):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'], num_labels=2
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=cw.to(self.device))
        optim = AdamW(model.parameters(),
                      lr=self.config['learning_rate'],
                      weight_decay=self.config['weight_decay'])
        total_steps = len(train_loader)*self.config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optim,
            int(total_steps*self.config['warmup_ratio']),
            total_steps
        )
        best_loss, patience = 1e9, 0
        for epoch in range(self.config['num_epochs']):
            model.train()
            tr_loss=0
            for b in train_loader:
                optim.zero_grad()
                out = model(
                    input_ids=b['input_ids'].to(self.device),
                    attention_mask=b['attention_mask'].to(self.device),
                    labels=b['labels'].to(self.device)
                )
                loss = criterion(out.logits, b['labels'].to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.config['max_grad_norm'])
                optim.step(); scheduler.step()
                tr_loss += loss.item()
            val_loss, val_acc, val_f1, val_auc = self.evaluate(model, val_loader, criterion)
            avg_tr = tr_loss/len(train_loader)
            print(f"Fold{fold+1} E{epoch+1}: TrL={avg_tr:.4f} VaL={val_loss:.4f} VA={val_acc:.4f} VF1={val_f1:.4f} VAUC={val_auc:.4f}")
            if val_loss<best_loss:
                best_loss,patience=val_loss,0
                torch.save(model.state_dict(),
                           f"{self.config['model_dir']}/fold{fold}.pt")
            else:
                patience+=1
                if patience>=self.config['early_stopping_patience']: break
        model.load_state_dict(torch.load(f"{self.config['model_dir']}/fold{fold}.pt"))
        return model,self.comprehensive_evaluate(model,val_loader)
    def evaluate(self, model, loader, crit):
        model.eval()
        losses, preds, labs, probs = [],[],[],[]
        with torch.no_grad():
            for b in loader:
                out = model(input_ids=b['input_ids'].to(self.device),
                            attention_mask=b['attention_mask'].to(self.device),
                            labels=b['labels'].to(self.device))
                losses.append(crit(out.logits, b['labels'].to(self.device)).item())
                p = torch.softmax(out.logits,1)
                preds.extend(torch.argmax(out.logits,1).cpu().numpy())
                labs.extend(b['labels'].numpy())
                probs.extend(p[:,1].cpu().numpy())
        vl,va,vf = np.mean(losses), accuracy_score(labs,preds), f1_score(labs,preds,average='weighted')
        try: auc = roc_auc_score(labs,probs)
        except: auc=0.0
        return vl,va,vf,auc
    def comprehensive_evaluate(self,model,loader):
        model.eval();p,l,pr=[],[],[]
        with torch.no_grad():
            for b in loader:
                o = model(input_ids=b['input_ids'].to(self.device),
                          attention_mask=b['attention_mask'].to(self.device))
                p.extend(torch.argmax(o.logits,1).cpu().numpy())
                l.extend(b['labels'].numpy())
                pr.extend(torch.softmax(o.logits,1)[:,1].cpu().numpy())
        return {
            'accuracy':accuracy_score(l,p),
            'f1_score':f1_score(l,p,average='weighted'),
            'roc_auc':roc_auc_score(l,pr),
            'brier_score':brier_score_loss(l,pr)
        }
    def train_kfold(self,df):
        set_seed(CONFIG['random_seed'])
        X,y=df['processed_text'].values,df['label'].values
        cw=torch.tensor(compute_class_weight('balanced',classes=np.unique(y),y=y),dtype=torch.float)
        skf=StratifiedKFold(n_splits=self.config['n_folds'],shuffle=True,random_state=CONFIG['random_seed'])
        models,metrics=[],[]
        for i,(tr,va) in enumerate(skf.split(X,y)):
            td,vl=list(X[tr]),list(y[tr]),list(X[tr]),list(y[tr])

            tr_ds=FakeNewsDataset(list(X[tr]),list(y[tr]),self.tokenizer,CONFIG['max_length'])
            va_ds=FakeNewsDataset(list(X[va]),list(y[va]),self.tokenizer,CONFIG['max_length'])
            tr_ld=DataLoader(tr_ds,batch_size=CONFIG['batch_size'],shuffle=True)
            va_ld=DataLoader(va_ds,batch_size=CONFIG['batch_size'],shuffle=False)
            m,mt=self.train_fold(tr_ld,va_ld,cw,i)
            models.append(m);metrics.append(mt)
        self.models,self.fold_metrics=models,metrics
        avg={k:np.mean([m[k] for m in metrics]) for k in metrics[0]}
        print("AVG METRICS:",avg)
        return avg
    def save_best_model(self):
        best=np.argmax([m['accuracy'] for m in self.fold_metrics])
        self.models[best].save_pretrained(f"{CONFIG['model_dir']}/best_model")
        self.tokenizer.save_pretrained(f"{CONFIG['model_dir']}/best_model")
    def load_best_model(self):
        mp=f"{CONFIG['model_dir']}/best_model"
        model=AutoModelForSequenceClassification.from_pretrained(mp).to(self.device)
        tok=AutoTokenizer.from_pretrained(mp)
        return model,tok
    def predict_single(self,text,model=None,tokenizer=None):
        if model is None:
            model,tokenizer=self.load_best_model()
        model.eval()
        txt=preprocess_text(text)
        enc=tokenizer.encode_plus(txt,add_special_tokens=True,
                                  max_length=CONFIG['max_length'],
                                  padding='max_length',truncation=True,
                                  return_tensors='pt')
        i,am=enc['input_ids'].to(self.device),enc['attention_mask'].to(self.device)
        with torch.no_grad():
            out=model(i,attention_mask=am)
            pr=torch.softmax(out.logits,1)
            p=torch.argmax(out.logits,1).cpu().item()
        fp, rp=pr[0][0].item(),pr[0][1].item()
        lbl="Fake" if p==0 else "Real"
        return {'text':text,'prediction':lbl,
                'confidence':max(fp,rp),
                'fake_probability':fp,
                'real_probability':rp}
    def interactive_mode(self):
        model,tok=self.load_best_model()
        st=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sf=f"{CONFIG['results_dir']}/session_{st}.txt"
        with open(sf,'w') as f:
            f.write(f"Session {st}\n")
        cnt=0
        print("Interactive mode: 'exit' to quit")
        while True:
            txt=input("Text: ").strip()
            if txt.lower()=='exit': break
            res=self.predict_single(txt,model,tok);cnt+=1
            print(res)
            with open(sf,'a') as f: f.write(json.dumps(res)+"\n")
        print(f"Logged {cnt} preds to {sf}")

def main():
    args = parse_args()

    # GPU selection
    gpus = torch.cuda.device_count()
    if gpus:
        choice = os.environ.get('CUDA_VISIBLE_DEVICES','')
        os.environ['CUDA_VISIBLE_DEVICES'] = choice

    # Apply hardware profile
    if args.profile == 'low':
        CONFIG.update(batch_size=8, max_length=128, num_epochs=3)
    else:
        CONFIG.update(batch_size=32, max_length=512, num_epochs=5)

    set_seed(CONFIG['random_seed'])
    detector = ProductionFakeNewsDetector(CONFIG)

    # Use args.csv or default
    csv_path = args.csv or CONFIG['default_csv']
    df = detector.load_data(csv_path)

    detector.train_kfold(df)
    detector.save_best_model()

    choice = input("Interactive? (y/n): ").strip().lower()
    if choice == 'y':
        detector.interactive_mode()

if __name__=="__main__":
    main()
