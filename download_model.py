#!/usr/bin/env python3
"""Download hand landmarker model from MediaPipe models repository."""

import urllib.request
import os
import sys

def download_model():
    """Download hand landmarker model."""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'hand_landmarker.task')
    
    # Try different URLs
    urls = [
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
        "https://github.com/google-ai-edge/mediapipe-models/releases/download/hand_landmarker/hand_landmarker.task",
    ]
    
    for url in urls:
        try:
            print(f"Downloading from: {url}")
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print("Could not download model. Please download manually from:")
    print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task")
    sys.exit(1)

if __name__ == "__main__":
    download_model()
