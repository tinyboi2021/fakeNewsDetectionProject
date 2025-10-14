#!/usr/bin/env python3
"""
Custom Clickbait Detection Model Training
"""

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

class ClickbaitTrainer:
    def __init__(self, model_name="roberta-base", output_dir="./clickbait_roberta"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_data(self, csv_path):
        """Load and prepare clickbait dataset"""
        df = pd.read_csv(csv_path)
        return df
    
    def prepare_dataset(self, df, test_size=0.2):
        """Prepare dataset for training"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)
        
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
    
    def train_model(self, train_dataset, val_dataset, num_epochs=3):
        """Train the clickbait detection model"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.to(self.device)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs/clickbait',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=-1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def main():
    trainer = ClickbaitTrainer()
    
    # Load your dataset
    df = trainer.load_data("/app/data/clickbait_dataset.csv")
    train_ds, val_ds = trainer.prepare_dataset(df)
    
    # Train model
    trainer.train_model(train_ds, val_ds, num_epochs=3)
    
    print("Clickbait model training completed!")

if __name__ == "__main__":
    main()