import cv2
import numpy as np
from ultralytics import YOLO

# Load pre-trained model (yolov8n is fast and lightweight)
# For specific defect detection, replace this with your custom trained weights (e.g., 'best.pt')
model = YOLO("yolov8n.pt")


def predict_objects(image_bytes: bytes) -> list:
    # Convert raw bytes to numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")

    # Run inference
    results = model(image)[0]

    detections = []
    # Parse results
    for box in results.boxes:
        # Get coordinates, confidence, and class ID
        xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = results.names[class_id]

        detections.append(
            {
                "label": label,
                "confidence": round(confidence, 2),
                "box": [int(coord) for coord in xyxy],
            }
        )

    return detections
