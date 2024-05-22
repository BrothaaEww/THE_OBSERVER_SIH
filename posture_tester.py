import cv2
import torch
import face_recognition
import pickle
import winsound
from gtts import gTTS
import pygame
from io import BytesIO

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
model.eval()

# Load COCO names file
with open("F:/Coding/Python_VS_Code/sec_proj/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load known faces and encodings
try:
    with open("F:/Coding/Python_VS_Code/sec_proj/cache/known_face_encoding.pkl", "rb") as f:
        known_face_encoding = pickle.load(f)
except FileNotFoundError:
    known_face_encoding = []

try:
    with open("F:/Coding/Python_VS_Code/sec_proj/cache/known_faces_names.pkl", "rb") as f:
        known_faces_names = pickle.load(f)
except FileNotFoundError:
        known_faces_names = []

# Function to generate speech for an unknown person
def generate_speech():
    text = "Unknown person detected!"
    tts = gTTS(text=text, lang="en", slow=False)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    pygame.init()
    pygame.mixer.music.load(mp3_fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Add a user to known faces
def add_user(name):
    user_image = face_recognition.load_image_file("F:/Coding/Python_VS_Code/sec_proj/photos/"+name.replace(" ", "_")+".jpg")
    user_encoding = face_recognition.face_encodings(user_image)[0]
    known_face_encoding.append(user_encoding)
    known_faces_names.append(name)
    with open("F:/Coding/Python_VS_Code/sec_proj/cache/known_face_encoding.pkl", "wb") as f:
        pickle.dump(known_face_encoding, f)
    with open("F:/Coding/Python_VS_Code/sec_proj/cache/known_faces_names.pkl", "wb") as f:
        pickle.dump(known_faces_names, f)

# Analyze posture based on bounding box coordinates
def analyze_posture(x, y, x_max, y_max, frame):
    box_width = x_max - x
    box_height = y_max - y

    aspect_ratio = box_height / box_width

    if aspect_ratio < 0.8:
        posture = "Sitting"
    elif aspect_ratio > 1.2:
        posture = "Standing"
    else:
        posture = "Unknown"

    cv2.putText(frame, f"Posture: {posture}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return aspect_ratio

# Detect and recognize persons, faces, and analyze posture
def detect_and_recognize_people():
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Detect and recognize persons and faces
        results = model(frame)
        for det in results.pred[0]:
            bbox = det[:4].int().cpu().numpy()
            x, y, x_max, y_max = bbox

            # If the detected object is a person (class index 0 is usually "person")
            if int(det[5]) == 0:
                # Convert to integer as OpenCV expects integers
                x, y, x_max, y_max = int(x), int(y), int(x_max), int(y_max)
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)

                # Analyze posture based on bounding box coordinates
                analyze_posture(x, y, x_max, y_max, frame)

        # Resize the frame to a larger size
        frame_resized = cv2.resize(frame, (1280, 850))  # Adjust dimensions as needed

        cv2.imshow('Human Detection and Facial Recognition', frame_resized)

        # Check for key events
        key = cv2.waitKey(1)

        # Close window if the user presses the 'q' key or the close button (X)
        if key == ord('q') or key == 27:  # 'q' or escape key
            break
        
        if cv2.getWindowProperty('Human Detection and Facial Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break
        
    video.release()
    cv2.destroyAllWindows()

# Run the function to detect and recognize people
detect_and_recognize_people()
