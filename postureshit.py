import cv2
import torch
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can change the model size (e.g., 'yolov5m', 'yolov5l', 'yolov5x')

# Set the device (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Load the COCO classes for YOLOv5
coco_classes = model.names

# Define classes related to violence detection (you may need to modify this based on your use case)
violence_classes = ['person', 'gun', 'knife']  # Example: person, gun, knife

# Function to detect violence in a frame
def detect_violence(frame):
    # Perform object detection
    results = model(frame)

    # Process the detections
    for det in results.pred[0]:
        class_idx = int(det[5])
        confidence = float(det[4])
        class_name = coco_classes[class_idx]

        # Check if the detected object is related to violence
        if class_name in violence_classes and confidence > 0.5:
            return True

    return False

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Perform violence detection
    is_violence_detected = detect_violence(frame)

    # Display the result
    if is_violence_detected:
        cv2.putText(frame, "Violence Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Violence Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
