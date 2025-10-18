import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml

# with open('C:/Users/paul-/Documents/GitHub/ClashRoyaleBuildABot/clashroyalebuildabot/detectors/unit_detector/dataset/data.yaml', 'r') as file:
#    data = yaml.safe_load(file)
#    print(data['names'])


def train_yolov8():
    """Train YOLOv8 model - simplest approach"""

    # Load a pre-trained model
    # model = YOLO('yolov8n.pt')  # yolov8n.pt, yolov8s.pt, yolov8m.pt etc.
    model = YOLO(
        "C:/Users/paul-/Documents/GitHub/ClashRoyaleBuildABot/runs/detect/train16/weights/last.pt"
    )

    # Train the model
    model.train(
        data="C:/Users/paul-/Documents/GitHub/ClashRoyaleBuildABot/clashroyalebuildabot/detectors/unit_detector/dataset/data.yaml",  # dataset config file
        epochs=30,
        imgsz=640,
        batch=-1,
        lr0=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        workers=4,
        patience=10,  # early stopping patience
        save=True,
        pretrained=True,
    )

    return model


if __name__ == "__main__":
    model = train_yolov8()
