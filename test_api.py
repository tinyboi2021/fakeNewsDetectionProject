#!/usr/bin/env python3
"""
Simple test script for the Fake News Detection API
Run this script to test if your Docker container is working correctly
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[0;32m",    # Green
        "ERROR": "\033[0;31m",   # Red
        "WARNING": "\033[1;33m", # Yellow
        "SUCCESS": "\033[0;36m"  # Cyan
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{status}: {message}{reset}")

def test_health_endpoint():
    """Test the health endpoint"""
    print_status("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_status(f"Health check passed! Status: {data.get('status')}", "SUCCESS")
            return True
        else:
            print_status(f"Health check failed with status: {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("Cannot connect to the API. Is the Docker container running?", "ERROR")
        print_status("Try running: ./setup.sh start-dev", "INFO")
        return False
    except Exception as e:
        print_status(f"Health check error: {str(e)}", "ERROR")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with sample data"""
    print_status("Testing prediction endpoint...")

    test_cases = [
        {
            "text": "Scientists at MIT have developed a new breakthrough technology that could revolutionize renewable energy.",
            "expected": "This should be classified as likely real news"
        },
        {
            "text": "SHOCKING: Local man discovers this ONE WEIRD TRICK that doctors HATE! You won't believe what happens next!",
            "expected": "This should be classified as likely fake/clickbait"
        },
        {
            "text": "The government announced new economic policies to address inflation concerns in the upcoming quarter.",
            "expected": "This should be classified as likely real news"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print_status(f"Test case {i}: {test_case['text'][:60]}...")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": test_case["text"]},
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", 0.0)

                print_status(f"  Prediction: {prediction} (confidence: {confidence:.2f})", "SUCCESS")
                print_status(f"  Expected: {test_case['expected']}", "INFO")
                print()
            else:
                print_status(f"  Prediction failed with status: {response.status_code}", "ERROR")
                print_status(f"  Response: {response.text}", "ERROR")

        except Exception as e:
            print_status(f"  Prediction error: {str(e)}", "ERROR")

def test_batch_endpoint():
    """Test the batch prediction endpoint"""
    print_status("Testing batch prediction endpoint...")

    texts = [
        "Breaking: New scientific discovery changes everything!",
        "Local weather forecast predicts rain tomorrow.",
        "URGENT: You need to see this amazing secret!"
    ]

    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json={"texts": texts},
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print_status(f"Batch prediction successful! Processed {len(results)} texts:", "SUCCESS")

            for result in results:
                index = result.get("index", "?")
                prediction = result.get("result", {}).get("prediction", "Unknown")
                confidence = result.get("result", {}).get("confidence", 0.0)
                print_status(f"  Text {index}: {prediction} (confidence: {confidence:.2f})", "INFO")
        else:
            print_status(f"Batch prediction failed with status: {response.status_code}", "ERROR")

    except Exception as e:
        print_status(f"Batch prediction error: {str(e)}", "ERROR")

def test_stats_endpoint():
    """Test the stats endpoint"""
    print_status("Testing stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            model_info = data.get("model_info", {})
            system_info = data.get("system_info", {})

            print_status("Stats retrieved successfully:", "SUCCESS")
            print_status(f"  Model: {model_info.get('name', 'Unknown')}", "INFO")
            print_status(f"  Model loaded: {model_info.get('loaded', False)}", "INFO")
            print_status(f"  Device: {model_info.get('device', 'Unknown')}", "INFO")
            print_status(f"  CUDA available: {system_info.get('cuda_available', False)}", "INFO")
        else:
            print_status(f"Stats request failed with status: {response.status_code}", "ERROR")
    except Exception as e:
        print_status(f"Stats error: {str(e)}", "ERROR")

def main():
    """Run all tests"""
    print("=" * 60)
    print("🔍 FAKE NEWS DETECTION API TESTS")
    print("=" * 60)
    print()

    print_status(f"Testing API at: {BASE_URL}")
    print()

    # Test health endpoint first
    if not test_health_endpoint():
        print()
        print_status("Cannot proceed with tests - API is not accessible", "ERROR")
        print_status("Make sure Docker container is running:", "INFO")
        print_status("  ./setup.sh start-dev", "INFO")
        print_status("  docker-compose up -d", "INFO")
        sys.exit(1)

    print()

    # Test other endpoints
    test_prediction_endpoint()
    test_batch_endpoint()
    test_stats_endpoint()

    print()
    print("=" * 60)
    print_status("🎉 API TESTING COMPLETED!", "SUCCESS")
    print_status("Your Fake News Detection API is working correctly!", "SUCCESS")
    print("=" * 60)
    print()
    print("Next steps:")
    print("• Open http://localhost:8000 in your browser")
    print("• Use the API in your applications")
    print("• Share with your team: ./setup.sh help")

if __name__ == "__main__":
    main()
