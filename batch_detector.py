
"""
Batch Fake News Detection Script
===============================

Process multiple news articles from a CSV file and save results.

Usage:
python batch_detector.py input_file.csv output_file.csv

CSV format should have columns: title, text
Results will include: title, text, prediction, confidence
"""

import pandas as pd
import sys
from quick_start_detector import QuickFakeNewsDetector

def process_csv(input_file, output_file):
    """
    Process a CSV file with news articles.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    # Load detector
    detector = QuickFakeNewsDetector()
    detector.load_model('roberta_winterforest')

    if not detector.classifier:
        print("Could not load model. Exiting.")
        return

    # Read input file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} articles from {input_file}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Process articles
    results = []

    for idx, row in df.iterrows():
        title = str(row.get('title', ''))
        text = str(row.get('text', ''))

        # Combine title and text
        full_text = f"{title} {text}".strip()

        if full_text:
            result = detector.predict(full_text)
            if result:
                results.append({
                    'title': title,
                    'text': text,
                    'prediction': result['prediction'],
                    'confidence': result['confidence']
                })
            else:
                results.append({
                    'title': title,
                    'text': text,
                    'prediction': 'ERROR',
                    'confidence': 0.0
                })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} articles...")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print summary
    fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
    real_count = sum(1 for r in results if r['prediction'] == 'REAL')
    error_count = sum(1 for r in results if r['prediction'] == 'ERROR')

    print(f"\nSummary:")
    print(f"- Real news: {real_count}")
    print(f"- Fake news: {fake_count}")
    print(f"- Errors: {error_count}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_detector.py input_file.csv output_file.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_csv(input_file, output_file)

if __name__ == "__main__":
    main()
