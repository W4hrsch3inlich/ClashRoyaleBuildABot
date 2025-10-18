import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml


def plot_bboxes(results):
    """
    Plots bounding boxes and labels on the image from a YOLO results object.
    Adapted from a practical example:cite[5].
    """
    # Extract data from the first result in the batch
    img = results[0].orig_img  # Original image as a numpy array
    names = results[0].names  # Dictionary of class names
    boxes = results[0].boxes  # Boxes object

    # Extract tensors to CPU and convert to numpy arrays
    scores = boxes.conf.cpu().numpy()  # Confidence scores
    classes = boxes.cls.cpu().numpy().astype(int)  # Class indices
    bboxes = (
        boxes.xyxy.cpu().numpy().astype(np.int32)
    )  # Bounding boxes in [x1, y1, x2, y2] format

    # Loop through all detected objects
    for score, cls, bbox in zip(scores, classes, bboxes):
        class_label = names[cls]  # Get class label from dictionary
        label = (
            f"{class_label}: {score:.2f}"  # Create label with class and score
        )

        # Draw the bounding box rectangle
        cv2.rectangle(
            img,
            (bbox[0], bbox[1]),  # Top-left corner
            (bbox[2], bbox[3]),  # Bottom-right corner
            color=(0, 255, 0),  # Green color (BGR format)
            thickness=2,
        )

        # Calculate text background for better readability
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        y_label_position = max(
            bbox[1] - 10, label_height + 10
        )  # Ensure label stays inside image top

        # Draw background for the text
        cv2.rectangle(
            img,
            (bbox[0], y_label_position - label_height - 10),
            (bbox[0] + label_width + 10, y_label_position + 5),
            color=(0, 255, 0),
            thickness=-1,
        )  # Filled rectangle

        # Put the label text on the image
        cv2.putText(
            img,
            label,
            (bbox[0] + 5, y_label_position - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Font scale
            (255, 255, 255),  # White color
            2,
        )  # Thickness

    return img


model = YOLO(
    "C:/Users/paul-/Documents/GitHub/ClashRoyaleBuildABot/runs/detect/train15/weights/best.pt"
)
results = model.predict(
    "C:/Users/paul-/Documents/GitHub/ClashRoyaleBuildABot/clashroyalebuildabot/detectors/unit_detector/dataset/test/images/019_png.rf.f964cf8e9e1a0e5aa4beee21b70f417b.jpg",
    conf=0.25,
)  # Run inference

# Generate the image with bounding boxes
annotated_image = plot_bboxes(results)

# Display the image (requires a GUI environment)
cv2.imshow("Training Feedback", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
