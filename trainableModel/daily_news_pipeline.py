#!/usr/bin/env python3
"""
Enhanced Daily News Pipeline with Advanced Model Training
- Integrates with production fake news detector
- Incremental training with all advanced features
- Comprehensive visualization and logging
- Automatic model updates
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import feedparser
import schedule
import time
import logging
from typing import List, Dict
import sqlite3
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== CONFIGURATION =====================
CONFIG = {
    'model_path': './production_model/best_model',
    'db_path': './news_database.db',
    'log_path': './logs/daily_update.log',
    'plots_dir': './plots_daily',
    'batch_size': 8,
    'accumulation_steps': 4,
    'update_threshold': 100,
    'confidence_threshold': 0.7,
    'training_lr': 1e-6,
    'training_epochs': 2,
    'backup_days': 7,
    'max_length': 384,
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
}

NEWS_SOURCES = {
    'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
    'reuters': 'https://www.reutersagency.com/feed/',
    'cnn': 'http://rss.cnn.com/rss/cnn_topstories.rss',
    'nytimes': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'guardian': 'https://www.theguardian.com/world/rss',
}

FACTCHECK_APIS = {
    'google_factcheck': 'https://factchecktools.googleapis.com/v1alpha1/claims:search',
}

# ===================== LOGGING SETUP =====================
os.makedirs('logs', exist_ok=True)
os.makedirs(CONFIG['plots_dir'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_path']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== DATABASE =====================
class NewsDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                url TEXT UNIQUE,
                published_date DATETIME,
                scraped_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                label INTEGER,
                confidence FLOAT,
                verified BOOLEAN DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                num_samples INTEGER,
                accuracy FLOAT,
                f1_score FLOAT,
                roc_auc FLOAT,
                model_path TEXT
            )
        ''')
        
        self.conn.commit()
    
    def insert_article(self, article: Dict):
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO news_articles (title, content, source, url, published_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['content'],
                article['source'],
                article['url'],
                article['published_date']
            ))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
    
    def update_prediction(self, article_id: int, label: int, confidence: float):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE news_articles 
            SET label = ?, confidence = ?
            WHERE id = ?
        ''', (label, confidence, article_id))
        self.conn.commit()
    
    def get_unlabeled_articles(self, limit=200):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, content, confidence
            FROM news_articles
            WHERE verified = 0
            ORDER BY scraped_date DESC
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()
    
    def get_uncertain_articles(self, limit=100):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, content, confidence
            FROM news_articles
            WHERE verified = 0 AND confidence < ?
            ORDER BY scraped_date DESC
            LIMIT ?
        ''', (CONFIG['confidence_threshold'], limit))
        return cursor.fetchall()
    
    def get_verified_articles(self, days=7):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT title, content, label
            FROM news_articles
            WHERE verified = 1 
            AND scraped_date > datetime('now', '-' || ? || ' days')
        ''', (days,))
        return cursor.fetchall()
    
    def mark_verified(self, article_ids: List[int]):
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(article_ids))
        cursor.execute(f'''
            UPDATE news_articles 
            SET verified = 1 
            WHERE id IN ({placeholders})
        ''', article_ids)
        self.conn.commit()
    
    def log_model_update(self, num_samples, accuracy, f1, auc, model_path):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO model_updates (num_samples, accuracy, f1_score, roc_auc, model_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (num_samples, accuracy, f1, auc, model_path))
        self.conn.commit()

# ===================== NEWS SCRAPER =====================
class NewsScraperPipeline:
    def __init__(self, db: NewsDatabase):
        self.db = db
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_rss_feed(self, feed_url: str, source_name: str) -> List[Dict]:
        articles = []
        try:
            feed = feedparser.parse(feed_url)
            logger.info(f"📰 Scraping {source_name}: {len(feed.entries)} articles")
            
            for entry in feed.entries[:20]:  # Limit to 20 per source
                article = {
                    'title': entry.get('title', '')[:500],
                    'content': self.extract_content(entry.get('link', '')),
                    'source': source_name,
                    'url': entry.get('link', ''),
                    'published_date': self.parse_date(entry.get('published', ''))
                }
                
                if article['content'] and len(article['content']) > 50:
                    articles.append(article)
        
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
        
        return articles
    
    def extract_content(self, url: str) -> str:
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
                tag.decompose()
            
            selectors = ['article', '.article-body', '.story-body']
            
            for selector in selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    paragraphs = content_div.find_all('p')
                    content = ' '.join([p.get_text().strip() for p in paragraphs])
                    return content[:5000]
            
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs[:10]])
            return content[:5000]
        
        except Exception as e:
            return ""
    
    def parse_date(self, date_str: str):
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return datetime.now()
    
    def scrape_all_sources(self) -> int:
        total_new = 0
        
        for source_name, feed_url in NEWS_SOURCES.items():
            articles = self.scrape_rss_feed(feed_url, source_name)
            
            for article in articles:
                article_id = self.db.insert_article(article)
                if article_id:
                    total_new += 1
        
        logger.info(f"✅ Scraped {total_new} new articles")
        return total_new

# ===================== ENHANCED MODEL =====================
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

class EnhancedNewsModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        enc = self.tokenizer.encode_plus(
            str(self.texts[i]), add_special_tokens=True,
            max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[i], dtype=torch.long)
        }

# ===================== INCREMENTAL TRAINER =====================
class IncrementalTrainer:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            checkpoint = torch.load(f"{model_path}/full_model.pt", map_location=self.device)
            self.model = EnhancedNewsModel(model_path).to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            logger.info(f"🤖 Loaded enhanced model from {model_path}")
        except:
            # Fallback to basic model
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            logger.info(f"🤖 Loaded basic model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if CONFIG['use_focal_loss']:
            self.loss_fn = FocalLoss(CONFIG['focal_alpha'], CONFIG['focal_gamma'])
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def predict_batch(self, articles: List[str]) -> List[tuple]:
        self.model.eval()
        predictions = []
        
        batch_size = CONFIG['batch_size']
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch, truncation=True, max_length=CONFIG['max_length'],
                padding=True, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                if isinstance(self.model, EnhancedNewsModel):
                    logits = self.model(inputs['input_ids'], inputs['attention_mask'])
                else:
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                for pred, prob in zip(preds, probs):
                    predictions.append((pred.item(), prob[pred].item()))
        
        return predictions
    
    def incremental_train(self, train_df: pd.DataFrame):
        logger.info(f"🎓 Starting incremental training on {len(train_df)} samples")
        
        texts = train_df['content'].tolist()
        labels = train_df['label'].tolist()
        
        dataset = NewsDataset(texts, labels, self.tokenizer, CONFIG['max_length'])
        loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=CONFIG['training_lr'])
        
        total_steps = len(loader) * CONFIG['training_epochs']
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        self.model.train()
        optimizer.zero_grad()
        
        for epoch in range(CONFIG['training_epochs']):
            total_loss = 0
            all_preds, all_labels = [], []
            
            for step, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if isinstance(self.model, EnhancedNewsModel):
                    logits = self.model(input_ids, attention_mask)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                
                loss = self.loss_fn(logits, labels) / CONFIG['accumulation_steps']
                loss.backward()
                
                if (step + 1) % CONFIG['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * CONFIG['accumulation_steps']
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            logger.info(f"  Epoch {epoch+1}/{CONFIG['training_epochs']}: "
                       f"Loss={total_loss/len(loader):.4f}, Acc={acc:.4f}, F1={f1:.4f}")
        
        logger.info("✅ Incremental training complete")
        return acc, f1
    
    def save_checkpoint(self, save_path: str):
        if isinstance(self.model, EnhancedNewsModel):
            self.model.encoder.save_pretrained(save_path)
            torch.save({
                'model_state': self.model.state_dict(),
                'config': CONFIG
            }, f"{save_path}/full_model.pt")
        else:
            self.model.save_pretrained(save_path)
        
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"💾 Model saved to {save_path}")

def plot_update_metrics(train_df, preds, probs):
    """Save visualization of model update"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(train_df['label'], preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Model Update {timestamp}\nConfusion Matrix')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    
    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(train_df['label'], probs)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_title(f'Model Update {timestamp}\nROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plots_dir']}/update_{timestamp}.png", dpi=150)
    plt.close()

# ===================== DAILY PIPELINE =====================
class DailyUpdatePipeline:
    def __init__(self):
        self.db = NewsDatabase(CONFIG['db_path'])
        self.scraper = NewsScraperPipeline(self.db)
        self.trainer = IncrementalTrainer(CONFIG['model_path'])
    
    def run_daily_cycle(self):
        logger.info("="*60)
        logger.info(f"🚀 Starting daily update cycle: {datetime.now()}")
        logger.info("="*60)
        
        # Step 1: Scrape
        num_scraped = self.scraper.scrape_all_sources()
        
        # Step 2: Get unlabeled
        unlabeled = self.db.get_unlabeled_articles(limit=200)
        logger.info(f"📊 Found {len(unlabeled)} unlabeled articles")
        
        # Step 3: Predict
        if unlabeled:
            contents = [content for _, _, content, _ in unlabeled]
            predictions = self.trainer.predict_batch(contents)
            
            for (article_id, _, _, _), (pred, conf) in zip(unlabeled, predictions):
                self.db.update_prediction(article_id, pred, conf)
            
            logger.info(f"✅ Made predictions on {len(predictions)} articles")
        
        # Step 4: Check for retraining
        verified = self.db.get_verified_articles(days=7)
        
        if len(verified) >= CONFIG['update_threshold']:
            logger.info(f"🎓 Sufficient verified data ({len(verified)} samples), retraining...")
            
            train_df = pd.DataFrame(verified, columns=['title', 'content', 'label'])
            
            acc, f1 = self.trainer.incremental_train(train_df)
            
            # Evaluate
            dataset = NewsDataset(
                train_df['content'].tolist(),
                train_df['label'].tolist(),
                self.trainer.tokenizer,
                CONFIG['max_length']
            )
            loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
            
            self.trainer.model.eval()
            all_preds, all_probs = [], []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch['input_ids'].to(self.trainer.device)
                    attention_mask = batch['attention_mask'].to(self.trainer.device)
                    
                    if isinstance(self.trainer.model, EnhancedNewsModel):
                        logits = self.trainer.model(input_ids, attention_mask)
                    else:
                        outputs = self.trainer.model(input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                    
                    probs = torch.softmax(logits, 1)
                    all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
            
            roc_auc = roc_auc_score(train_df['label'], all_probs)
            
            # Plot
            plot_update_metrics(train_df, all_preds, all_probs)
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"{CONFIG['model_path']}_checkpoint_{timestamp}"
            self.trainer.save_checkpoint(checkpoint_path)
            
            # Update main model
            self.trainer.save_checkpoint(CONFIG['model_path'])
            
            # Log to database
            self.db.log_model_update(len(verified), acc, f1, roc_auc, checkpoint_path)
            
            logger.info(f"✅ Model updated: Acc={acc:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
        else:
            logger.info(f"📊 Not enough verified data yet ({len(verified)}/{CONFIG['update_threshold']})")
        
        logger.info("✅ Daily cycle complete")
        logger.info("="*60)

# ===================== SCHEDULER =====================
def schedule_daily_updates():
    pipeline = DailyUpdatePipeline()
    
    schedule.every().day.at("02:00").do(pipeline.run_daily_cycle)
    
    logger.info("⏰ Scheduler started. Waiting for scheduled runs...")
    logger.info("   Next run: 02:00 AM daily")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def create_labeling_interface():
    db = NewsDatabase(CONFIG['db_path'])
    
    while True:
        articles = db.get_uncertain_articles(limit=10)
        
        if not articles:
            print("✅ No articles pending labeling!")
            break
        
        for idx, (article_id, title, content, confidence) in enumerate(articles, 1):
            print(f"\n{'='*60}")
            print(f"Article {idx}/{len(articles)} (ID: {article_id})")
            print(f"Confidence: {confidence:.2%}")
            print(f"{'='*60}")
            print(f"Title: {title}")
            print(f"\nContent:\n{content[:500]}...")
            print(f"\n{'='*60}")
            
            label = input("Label (0=Fake, 1=Real, s=Skip, q=Quit): ").strip().lower()
            
            if label == 'q':
                return
            elif label == 's':
                continue
            elif label in ['0', '1']:
                db.update_prediction(article_id, int(label), 1.0)
                db.mark_verified([article_id])
                print(f"✅ Labeled as {'FAKE' if label == '0' else 'REAL'}")

# ===================== MAIN =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['schedule', 'run-once', 'label'], 
                       default='schedule')
    args = parser.parse_args()
    
    if args.mode == 'schedule':
        schedule_daily_updates()
    elif args.mode == 'run-once':
        pipeline = DailyUpdatePipeline()
        pipeline.run_daily_cycle()
    elif args.mode == 'label':
        create_labeling_interface()
