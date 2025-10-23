# 📊 Dataset Combination Script - Usage Guide

## 🚀 Quick Start

### **Basic Usage**
```bash
python combine_clickbait_datasets.py
```

### **With Custom Paths**
```bash
python combine_clickbait_datasets.py \
  ./data/clickbait_train.csv \
  ./data/clickbait_test.csv \
  ./data/clickbait_combined.csv \
  undersample
```

---

## 📝 **What This Script Does**

### **1. Loads Both Datasets**
- ✅ Automatic encoding detection
- ✅ Error handling for missing files
- ✅ Supports multiple column names

### **2. Combines Train + Test**
```python
Train: 15,000 samples
Test:   5,000 samples
────────────────────
Combined: 20,000 samples
```

### **3. Cleans Data**
- ✅ Removes URLs
- ✅ Removes HTML tags
- ✅ Removes special characters
- ✅ Removes duplicates
- ✅ Removes null values
- ✅ Filters by text length

### **4. Balances Classes**
- ✅ Analyzes imbalance ratio
- ✅ Undersamples OR oversamples
- ✅ Shuffles data

### **5. Saves & Backs Up**
- ✅ Creates timestamped backups
- ✅ Saves to clean CSV
- ✅ Generates detailed logs

---

## 🎯 **Configuration Options**

Edit the `CONFIG` dictionary in the script:

```python
CONFIG = {
    # Input files (modify these!)
    'train_path': './data/clickbait_train.csv',
    'test_path': './data/clickbait_test.csv',
    
    # Output file
    'output_path': './data/clickbait_combined.csv',
    
    # Text filtering
    'min_text_length': 20,      # Too short = removed
    'max_text_length': 5000,    # Too long = removed
    
    # Cleaning options
    'remove_duplicates': True,
    'remove_null': True,
    
    # Balancing method
    'balance_method': 'undersample',  # 'undersample', 'oversample', 'none'
    
    # Random seed for reproducibility
    'random_seed': 42,
}
```

---

## 💾 **Output Files**

### **Main Output**
```
./data/clickbait_combined.csv
├── content (text)
├── label (0 or 1)
└── source (train or test)
```

### **Backup Files**
```
./data/backup/
├── clickbait_combined_backup_20251023_063000.csv
├── clickbait_combined_backup_20251023_093000.csv
└── ... (timestamped backups)
```

---

## 📊 **Example Output**

```
[2025-10-23 06:30:00] INFO: ======================================================================
[2025-10-23 06:30:00] INFO: 🚀 CLICKBAIT DATASET COMBINATION & CLEANING
[2025-10-23 06:30:00] INFO: ======================================================================

[2025-10-23 06:30:00] INFO: 📂 STEP 1: Loading Datasets
[2025-10-23 06:30:00] SUCCESS: ✅ Loaded ./data/clickbait_train.csv with encoding: utf-8
[2025-10-23 06:30:01] SUCCESS: ✅ Loaded ./data/clickbait_test.csv with encoding: utf-8

[2025-10-23 06:30:01] INFO: 🔗 STEP 2: Combining Datasets
[2025-10-23 06:30:01] INFO:    Train dataset: 15234 samples
[2025-10-23 06:30:01] INFO:    Test dataset: 5123 samples
[2025-10-23 06:30:01] SUCCESS: ✅ Using content column: 'content'
[2025-10-23 06:30:01] SUCCESS: ✅ Using label column: 'is_fakenews'
[2025-10-23 06:30:01] SUCCESS: 📊 Combined dataset: 20357 samples

[2025-10-23 06:30:02] INFO: 🧹 STEP 3: Cleaning Data
[2025-10-23 06:30:02] INFO: 🧹 Starting data cleaning...
[2025-10-23 06:30:02] INFO:    Initial: 20357 samples
[2025-10-23 06:30:02] INFO:    After removing nulls: 20234 samples (-123)
[2025-10-23 06:30:03] INFO:    After text filtering: 20001 samples (-233)
[2025-10-23 06:30:04] INFO:    After removing duplicates: 19876 samples (-125)
[2025-10-23 06:30:04] SUCCESS: ✅ Cleaning complete! Final: 19876 samples

[2025-10-23 06:30:04] INFO: 📊 STEP 4: Analyzing Class Balance
[2025-10-23 06:30:04] INFO: 📊 CLASS BALANCE ANALYSIS:
[2025-10-23 06:30:04] INFO:    Class 0 (Non-Clickbait):  14910 ( 75.0%)
[2025-10-23 06:30:04] INFO:    Class 1 (Clickbait):       4966 ( 25.0%)
[2025-10-23 06:30:04] INFO:    Imbalance Ratio:         3.00:1
[2025-10-23 06:30:04] INFO:    Balanced:                ❌ NO

[2025-10-23 06:30:04] INFO: ⚖️  STEP 5: Balancing Dataset
[2025-10-23 06:30:04] INFO: ⚖️  BALANCING DATASET (undersample):
[2025-10-23 06:30:04] INFO:    Before - Class 0: 14910, Class 1: 4966
[2025-10-23 06:30:05] INFO:    After  - Class 0: 4966, Class 1: 4966
[2025-10-23 06:30:05] SUCCESS: ✅ Balancing complete!

[2025-10-23 06:30:05] INFO: ✅ STEP 6: Verifying Balance
[2025-10-23 06:30:05] INFO: 📊 CLASS BALANCE ANALYSIS:
[2025-10-23 06:30:05] INFO:    Class 0 (Non-Clickbait):   4966 ( 50.0%)
[2025-10-23 06:30:05] INFO:    Class 1 (Clickbait):       4966 ( 50.0%)
[2025-10-23 06:30:05] INFO:    Imbalance Ratio:         1.00:1
[2025-10-23 06:30:05] INFO:    Balanced:                ✅ YES

[2025-10-23 06:30:05] INFO: 📈 STEP 7: Dataset Statistics
[2025-10-23 06:30:05] INFO: 📈 DATASET STATISTICS:
[2025-10-23 06:30:05] INFO:    Total samples: 9932
[2025-10-23 06:30:05] INFO:    Unique content: 9876
[2025-10-23 06:30:05] INFO:    Average text length: 342 chars
[2025-10-23 06:30:05] INFO:    Min text length: 21 chars
[2025-10-23 06:30:05] INFO:    Max text length: 4987 chars
[2025-10-23 06:30:05] INFO: 📂 DATASET SOURCES:
[2025-10-23 06:30:05] INFO:    train: 7234 samples
[2025-10-23 06:30:05] INFO:    test: 2698 samples

[2025-10-23 06:30:05] INFO: 💾 STEP 8: Saving Dataset
[2025-10-23 06:30:05] SUCCESS: 💾 Backup created: ./data/backup/clickbait_combined_backup_20251023_063005.csv
[2025-10-23 06:30:05] SUCCESS: 💾 Dataset saved: ./data/clickbait_combined.csv

======================================================================
✅ PROCESSING COMPLETE!
======================================================================

📊 SUMMARY:
   Input train: 15234 samples
   Input test: 5123 samples
   Combined: 20357 samples
   After cleaning: 19876 samples
   After balancing: 9932 samples
   Output file: ./data/clickbait_combined.csv
   Backup directory: ./data/backup
```

---

## 🔧 **Customization Examples**

### **Example 1: Balance Using Oversampling Instead**
```python
CONFIG['balance_method'] = 'oversample'  # Double minority class
```

### **Example 2: No Balancing (Keep All Data)**
```python
CONFIG['balance_method'] = 'none'
```

### **Example 3: Stricter Text Filtering**
```python
CONFIG['min_text_length'] = 50    # Minimum 50 chars
CONFIG['max_text_length'] = 2000  # Maximum 2000 chars
```

### **Example 4: Skip Duplicate Removal**
```python
CONFIG['remove_duplicates'] = False
```

---

## 📋 **Input CSV Format**

Your CSV should have these columns:

### **Format 1: content + is_fakenews**
```
content,is_fakenews
"This is a fake news headline",0
"Another article here",1
```

### **Format 2: Any column names (auto-detected)**
```
title,label
"Some headline",0
"Another headline",1
```

The script automatically detects:
- Content columns: `content`, `title`, `text`, `headline`
- Label columns: `label`, `is_fakenews`, `is_clickbait`, `is_fake`

---

## ✅ **Pre-Execution Checklist**

- [ ] Both CSV files exist
- [ ] CSV files have correct format
- [ ] Output directory exists or will be created
- [ ] Sufficient disk space for backup
- [ ] Python 3.7+
- [ ] pandas installed (`pip install pandas scikit-learn`)

---

## 🚨 **Troubleshooting**

### **Issue: "File not found"**
```bash
# Check file paths
ls -la ./data/clickbait_train.csv
ls -la ./data/clickbait_test.csv

# Fix: Update CONFIG paths
```

### **Issue: "No columns match"**
```bash
# Check your column names
python -c "import pandas as pd; df = pd.read_csv('./data/clickbait_train.csv'); print(df.columns.tolist())"

# The script will tell you what it found
```

### **Issue: Encoding errors**
```bash
# Script auto-detects encoding, but if it fails:
# Try manually specifying in your editor or Excel, then re-save as UTF-8
```

### **Issue: Out of memory**
```bash
# Use oversampling instead (uses less memory)
CONFIG['balance_method'] = 'oversample'
```

---

## 🎯 **Next Steps After Combining**

### **Use Combined Dataset for Training**
```python
# In your train_clickbait_detector.py
python train_clickbait_detector_enhanced.py \
  --csv ./data/clickbait_combined.csv \
  --source-model ./production_model/best_model
```

### **Use for Analysis**
```python
import pandas as pd

df = pd.read_csv('./data/clickbait_combined.csv')
print(df.describe())
print(df['label'].value_counts())
```

---

## 📊 **Verification Script**

```python
import pandas as pd

# Load combined dataset
df = pd.read_csv('./data/clickbait_combined.csv')

print("✅ VERIFICATION:")
print(f"   Total samples: {len(df)}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Class distribution:\n{df['label'].value_counts()}")
print(f"   Missing values:\n{df.isnull().sum()}")
print(f"   Sample text lengths: min={df['content'].str.len().min()}, max={df['content'].str.len().max()}")
```

---

## 💡 **Tips**

1. **Always check output file** before using for training
2. **Keep backups** - they're timestamped automatically
3. **Test with small dataset first** - run on subset
4. **Monitor class ratio** - should be near 1:1 after balancing
5. **Use undersampling** for speed, **oversampling** for more data

---

**Ready to combine your datasets!** 🚀
