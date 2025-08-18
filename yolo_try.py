
import json
import os
import shutil
from ultralytics import YOLO
import torch
from PIL import Image
import yaml
import time

def train_yolo_model():
    """Train the YOLO model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = YOLO('yolov8n.pt')
    with open('dataset.yaml', 'r') as f:
        data = yaml.safe_load(f)
        print(data)
    time.sleep(10)
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=1920,
        batch=16,
        device=device,
        patience=10,
        project='yolo_training',
        name='experiment_1'
    )
    
    return results

def evaluate_model():
    """Evaluate the trained model"""
    best_model = YOLO('yolo_training/experiment_1/weights/best.pt')
    metrics = best_model.val(data='dataset.yaml')
    
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return best_model

# Main execution
if __name__ == "__main__":
    # Train model
    train_yolo_model()
    
    # Evaluate model
    model = evaluate_model()
    
    print("Training pipeline completed!")