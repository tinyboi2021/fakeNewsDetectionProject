#!/usr/bin/env python3
"""
Dataset Combination & Cleaning Script
Combines clickbait_train.csv and clickbait_test.csv into a balanced dataset
with proper preprocessing and class balance handling

FIXED VERSION - No variable shadowing bug
"""

import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

# ===================== CONFIGURATION =====================
CONFIG = {
    'train_path': '/app/src/dataset2/Clickbait_dataset/clickbait_train.csv',
    'test_path': '/app/src/dataset2/Clickbait_dataset/clickbait_test.csv',
    'output_path': '/app/src/dataset2/Clickbait_dataset/clickbait_combined.csv',
    'backup_path': '/app/src/dataset2/Clickbait_dataset/backup',
    
    # Cleaning settings
    'min_text_length': 20,      # Minimum characters
    'max_text_length': 5000,    # Maximum characters
    'remove_duplicates': True,
    'remove_null': True,
    
    # Balance settings
    'balance_method': 'undersample',  # 'undersample', 'oversample', or 'none'
    'random_seed': 42,
}

# ===================== LOGGER =====================
def log(message: str, level: str = "INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

# ===================== DATA LOADING =====================
def load_csv_safe(filepath: str) -> pd.DataFrame:
    """Load CSV with error handling"""
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                log(f"✅ Loaded {filepath} with encoding: {encoding}", "SUCCESS")
                return df
            except UnicodeDecodeError:
                continue
        
        # If all fail, try without encoding
        df = pd.read_csv(filepath)
        log(f"✅ Loaded {filepath}", "SUCCESS")
        return df
    
    except FileNotFoundError:
        log(f"❌ File not found: {filepath}", "ERROR")
        return None
    except Exception as e:
        log(f"❌ Error loading {filepath}: {e}", "ERROR")
        return None

# ===================== TEXT PREPROCESSING =====================
def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep only alphanumeric and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\'\"]', '', text)
    
    return text.strip()

def filter_text(text: str, min_len: int = 20, max_len: int = 5000) -> bool:
    """Check if text passes quality filters"""
    if pd.isna(text):
        return False
    
    text_len = len(str(text))
    return min_len <= text_len <= max_len

# ===================== DATASET COMBINATION =====================
def combine_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Combine train and test datasets"""
    
    log(f"📊 Train dataset: {len(train_df)} samples", "INFO")
    log(f"📊 Test dataset: {len(test_df)} samples", "INFO")
    
    # Standardize column names
    train_df = train_df.rename(columns=lambda x: x.strip().lower())
    test_df = test_df.rename(columns=lambda x: x.strip().lower())
    
    # Identify content and label columns
    content_cols = [col for col in train_df.columns if 'content' in col or 'title' in col or 'text' in col]
    label_cols = [col for col in train_df.columns if 'label' in col or 'fake' in col or 'clickbait' in col]
    
    if not content_cols or not label_cols:
        log(f"❌ Could not identify content/label columns", "ERROR")
        log(f"   Train columns: {train_df.columns.tolist()}", "ERROR")
        log(f"   Test columns: {test_df.columns.tolist()}", "ERROR")
        return None
    
    content_col = content_cols[0]
    label_col = label_cols[0]
    
    log(f"✅ Using content column: '{content_col}'", "SUCCESS")
    log(f"✅ Using label column: '{label_col}'", "SUCCESS")
    
    # Select only relevant columns
    train_df = train_df[[content_col, label_col]].copy()
    test_df = test_df[[content_col, label_col]].copy()
    
    # Standardize names
    train_df.columns = ['content', 'label']
    test_df.columns = ['content', 'label']
    
    # Add source column to track origin
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    
    # Combine
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    log(f"📊 Combined dataset: {len(combined_df)} samples", "SUCCESS")
    
    return combined_df

# ===================== DATA CLEANING =====================
def clean_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean and validate dataset"""
    
    initial_count = len(df)
    log(f"🧹 Starting data cleaning...", "INFO")
    log(f"   Initial: {initial_count} samples", "INFO")
    
    # Remove null values
    if config['remove_null']:
        df = df.dropna(subset=['content', 'label'])
        log(f"   After removing nulls: {len(df)} samples (-{initial_count - len(df)})", "INFO")
    
    # Clean text
    df['content'] = df['content'].apply(clean_text)
    
    # Filter by length
    df = df[df['content'].apply(
        lambda x: filter_text(x, config['min_text_length'], config['max_text_length'])
    )]
    log(f"   After text filtering: {len(df)} samples (-{initial_count - len(df)})", "INFO")
    
    # Remove duplicates
    if config['remove_duplicates']:
        initial = len(df)
        df = df.drop_duplicates(subset=['content'])
        log(f"   After removing duplicates: {len(df)} samples (-{initial - len(df)})", "INFO")
    
    # Ensure label is int
    df['label'] = df['label'].astype(int)
    
    # Validate labels
    unique_labels = df['label'].unique()
    if not all(label in [0, 1] for label in unique_labels):
        log(f"⚠️  Found unexpected labels: {unique_labels}", "WARNING")
        df = df[df['label'].isin([0, 1])]
    
    log(f"✅ Cleaning complete! Final: {len(df)} samples", "SUCCESS")
    
    return df

# ===================== CLASS BALANCE ANALYSIS =====================
def analyze_balance(df: pd.DataFrame) -> dict:
    """Analyze class balance"""
    
    counts = df['label'].value_counts().sort_index()
    total = len(df)
    
    analysis = {
        'total_samples': total,
        'class_0': counts.get(0, 0),
        'class_1': counts.get(1, 0),
        'class_0_pct': (counts.get(0, 0) / total * 100) if total > 0 else 0,
        'class_1_pct': (counts.get(1, 0) / total * 100) if total > 0 else 0,
        'ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf'),
        'is_balanced': abs(counts.get(0, 0) - counts.get(1, 0)) < total * 0.1,
    }
    
    log(f"\n📊 CLASS BALANCE ANALYSIS:", "INFO")
    log(f"   Class 0 (Non-Clickbait): {analysis['class_0']:6d} ({analysis['class_0_pct']:5.1f}%)", "INFO")
    log(f"   Class 1 (Clickbait):     {analysis['class_1']:6d} ({analysis['class_1_pct']:5.1f}%)", "INFO")
    log(f"   Imbalance Ratio:         {analysis['ratio']:.2f}:1", "INFO")
    log(f"   Balanced:                {'✅ YES' if analysis['is_balanced'] else '❌ NO'}", "INFO")
    
    return analysis

# ===================== CLASS BALANCING =====================
def balance_dataset(df: pd.DataFrame, method: str = 'undersample', seed: int = 42) -> pd.DataFrame:
    """Balance dataset"""
    
    from sklearn.utils import resample
    
    if method == 'none':
        log("⏭️  Skipping balancing", "INFO")
        return df
    
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    
    n_class_0 = len(class_0)
    n_class_1 = len(class_1)
    
    log(f"\n⚖️  BALANCING DATASET ({method}):", "INFO")
    log(f"   Before - Class 0: {n_class_0}, Class 1: {n_class_1}", "INFO")
    
    if method == 'undersample':
        # Undersample majority class
        minority_count = min(n_class_0, n_class_1)
        
        if n_class_0 > n_class_1:
            class_0 = resample(class_0, n_samples=minority_count, random_state=seed)
        else:
            class_1 = resample(class_1, n_samples=minority_count, random_state=seed)
    
    elif method == 'oversample':
        # Oversample minority class
        majority_count = max(n_class_0, n_class_1)
        
        if n_class_0 < n_class_1:
            class_0 = resample(class_0, n_samples=majority_count, random_state=seed)
        else:
            class_1 = resample(class_1, n_samples=majority_count, random_state=seed)
    
    balanced_df = pd.concat([class_0, class_1], ignore_index=True).sample(frac=1, random_state=seed)
    
    log(f"   After  - Class 0: {len(class_0)}, Class 1: {len(class_1)}", "INFO")
    log(f"✅ Balancing complete!", "SUCCESS")
    
    return balanced_df

# ===================== STATISTICS =====================
def print_statistics(df: pd.DataFrame):
    """Print detailed dataset statistics"""
    
    log(f"\n📈 DATASET STATISTICS:", "INFO")
    log(f"   Total samples: {len(df)}", "INFO")
    log(f"   Unique content: {df['content'].nunique()}", "INFO")
    log(f"   Average text length: {df['content'].str.len().mean():.0f} chars", "INFO")
    log(f"   Min text length: {df['content'].str.len().min()} chars", "INFO")
    log(f"   Max text length: {df['content'].str.len().max()} chars", "INFO")
    
    if 'source' in df.columns:
        log(f"\n📂 DATASET SOURCES:", "INFO")
        for source, count in df['source'].value_counts().items():
            log(f"   {source}: {count} samples", "INFO")

# ===================== BACKUP CREATION =====================
def create_backup_file(df: pd.DataFrame, backup_dir: str):
    """Create backup of original data - FIXED NAME"""
    
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    backup_file = os.path.join(backup_dir, f"clickbait_combined_backup_{timestamp}.csv")
    df.to_csv(backup_file, index=False)
    
    log(f"💾 Backup created: {backup_file}", "SUCCESS")

# ===================== SAVE DATASET =====================
def save_dataset(df: pd.DataFrame, output_path: str, should_backup: bool = True):
    """Save combined dataset - FIXED PARAMETER NAME"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # FIXED: Renamed parameter from 'create_backup' to 'should_backup'
    # to avoid shadowing the function name
    if should_backup:
        backup_dir = os.path.join(os.path.dirname(output_path), 'backup')
        create_backup_file(df, backup_dir)
    
    df.to_csv(output_path, index=False)
    log(f"💾 Dataset saved: {output_path}", "SUCCESS")

# ===================== MAIN =====================
def main():
    """Main function"""
    
    log("=" * 70, "INFO")
    log("🚀 CLICKBAIT DATASET COMBINATION & CLEANING", "INFO")
    log("=" * 70, "INFO")
    
    # Step 1: Load datasets
    log("\n📂 STEP 1: Loading Datasets", "INFO")
    train_df = load_csv_safe(CONFIG['train_path'])
    test_df = load_csv_safe(CONFIG['test_path'])
    
    if train_df is None or test_df is None:
        log("❌ Failed to load datasets", "ERROR")
        return False
    
    # Step 2: Combine datasets
    log("\n🔗 STEP 2: Combining Datasets", "INFO")
    combined_df = combine_datasets(train_df, test_df)
    if combined_df is None:
        return False
    
    # Step 3: Clean data
    log("\n🧹 STEP 3: Cleaning Data", "INFO")
    cleaned_df = clean_dataset(combined_df, CONFIG)
    
    # Step 4: Analyze balance
    log("\n📊 STEP 4: Analyzing Class Balance", "INFO")
    balance_before = analyze_balance(cleaned_df)
    
    # Step 5: Balance dataset
    log("\n⚖️  STEP 5: Balancing Dataset", "INFO")
    if balance_before['ratio'] > 2.0:  # If imbalanced
        balanced_df = balance_dataset(cleaned_df, CONFIG['balance_method'], CONFIG['random_seed'])
    else:
        log("✅ Dataset already balanced", "SUCCESS")
        balanced_df = cleaned_df
    
    # Step 6: Verify balance
    log("\n✅ STEP 6: Verifying Balance", "INFO")
    balance_after = analyze_balance(balanced_df)
    
    # Step 7: Statistics
    log("\n📈 STEP 7: Dataset Statistics", "INFO")
    print_statistics(balanced_df)
    
    # Step 8: Save dataset - FIXED CALL
    log("\n💾 STEP 8: Saving Dataset", "INFO")
    save_dataset(balanced_df, CONFIG['output_path'], should_backup=True)
    
    # Summary
    log("\n" + "=" * 70, "INFO")
    log("✅ PROCESSING COMPLETE!", "SUCCESS")
    log("=" * 70, "INFO")
    log(f"\n📊 SUMMARY:", "INFO")
    log(f"   Input train: {len(train_df)} samples", "INFO")
    log(f"   Input test: {len(test_df)} samples", "INFO")
    log(f"   Combined: {len(combined_df)} samples", "INFO")
    log(f"   After cleaning: {len(cleaned_df)} samples", "INFO")
    log(f"   After balancing: {len(balanced_df)} samples", "INFO")
    log(f"   Output file: {CONFIG['output_path']}", "INFO")
    log(f"   Backup directory: {os.path.join(os.path.dirname(CONFIG['output_path']), 'backup')}", "INFO")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Allow command-line configuration
    if len(sys.argv) > 1:
        CONFIG['train_path'] = sys.argv[1]
    if len(sys.argv) > 2:
        CONFIG['test_path'] = sys.argv[2]
    if len(sys.argv) > 3:
        CONFIG['output_path'] = sys.argv[3]
    if len(sys.argv) > 4:
        CONFIG['balance_method'] = sys.argv[4]
    
    success = main()
    sys.exit(0 if success else 1)
