# Fake News Detection using RoBERTa

A comprehensive fake news detection system using RoBERTa (Robustly Optimized BERT Pretraining Approach) for binary classification of news articles as fake or real.

## Features

- **State-of-the-art Model**: Uses RoBERTa for superior performance in text classification
- **Complete Pipeline**: Data preprocessing, model training, evaluation, and inference
- **Pre-trained Models**: Support for multiple pre-trained fake news detection models
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix
- **Batch Processing**: Predict multiple articles at once
- **Easy to Use**: Simple API for both training and inference

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd fake-news-detection-roberta
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The system supports various fake news datasets:

- **ISOT Fake News Dataset**: 44,898 articles (Real: 21,417, Fake: 23,481)
- **Kaggle Fake and Real News Dataset**: Combined dataset from multiple sources
- **Custom Datasets**: Support for any CSV format with title, text, and label columns

### Dataset Format

Your dataset should have the following columns:
- `title`: News article headline
- `text`: Full article content  
- `label`: 0 for fake news, 1 for real news

## Usage

### Basic Usage

```python
from fake_news_roberta_detector import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Load data
df = detector.load_data(fake_csv_path="Fake.csv", true_csv_path="True.csv")

# Prepare dataset
train_dataset, val_dataset = detector.prepare_dataset(df)

# Train model
trainer = detector.train_model(train_dataset, val_dataset)

# Evaluate model
results = detector.evaluate_model(val_dataset, trainer)

# Make predictions
prediction = detector.predict("Breaking: Scientists discover cure for all diseases!")
print(f"Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.3f})")
```

### Using Pre-trained Models

```python
from transformers import pipeline

# Load pre-trained model
classifier = pipeline("text-classification", 
                     model="hamzab/roberta-fake-news-classification")

# Make prediction
result = classifier("<title>Your news title<content>Your news content<end>")
print(f"Prediction: {result[0]['label']} (Score: {result[0]['score']:.3f})")
```

## Model Performance

Based on research and implementations:

- **RoBERTa-base**: 89.23% precision, 90.14% recall, 89.68% F1-score
- **Fine-tuned models**: Up to 99% accuracy on specific datasets
- **Training time**: 2-3 hours on GPU for full dataset

## Configuration

Modify `config.ini` to adjust:
- Model parameters (learning rate, batch size, epochs)
- Data preprocessing settings
- Evaluation options

## Project Structure

```
├── fake_news_roberta_detector.py    # Main implementation
├── requirements.txt                 # Dependencies
├── config.ini                      # Configuration settings
├── README.md                       # This file
└── output/                         # Saved models and results
    ├── fake_news_roberta/          # Trained model
    └── confusion_matrix.png        # Evaluation plots
```

## Advanced Features

### Custom Preprocessing

```python
detector = FakeNewsDetector()

# Custom preprocessing function
def custom_preprocess(text):
    # Your custom preprocessing logic
    return processed_text

# Apply custom preprocessing
detector.preprocess_text = custom_preprocess
```

### Hyperparameter Tuning

```python
# Train with custom parameters
trainer = detector.train_model(
    train_dataset, 
    val_dataset,
    num_epochs=5,
    learning_rate=1e-5,
    batch_size=32
)
```

### Batch Prediction

```python
news_articles = [
    "Article 1 content...",
    "Article 2 content...",
    "Article 3 content..."
]

predictions = detector.batch_predict(news_articles)
for pred in predictions:
    print(f"{pred['prediction']}: {pred['confidence']:.3f}")
```

## Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall correctness
- **Precision**: True positive rate for each class
- **Recall**: Sensitivity for each class  
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## Hardware Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB free space

### Recommended for Training
- GPU: NVIDIA with 8GB+ VRAM
- RAM: 16GB+
- CPU: 8+ cores

## Troubleshooting

### Out of Memory Error
- Reduce batch size in config.ini
- Use gradient accumulation
- Try DistilRoBERTa for lower memory usage

### Poor Performance
- Check data quality and preprocessing
- Increase training epochs
- Try different learning rates
- Use more training data

### CUDA Issues
- Install PyTorch with CUDA support
- Check GPU compatibility
- Verify CUDA version

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fakenews_roberta_2025,
  title={Fake News Detection using RoBERTa},
  author={AI Research Assistant},
  year={2025},
  url={https://github.com/your-repo/fake-news-roberta}
}
```

## References

1. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
2. Ahmed, H., et al. (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques.
3. Transformers: State-of-the-art Natural Language Processing. Hugging Face.

## Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the configuration options
