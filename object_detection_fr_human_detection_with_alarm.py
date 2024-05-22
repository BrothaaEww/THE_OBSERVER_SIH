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
known_face_encoding_file = "F:/Coding/Python_VS_Code/sec_proj/cache/known_face_encoding.pkl"
known_faces_names_file = "F:/Coding/Python_VS_Code/sec_proj/cache/known_faces_names.pkl"

try:
    with open(known_face_encoding_file, "rb") as f:
        known_face_encoding = pickle.load(f)
except FileNotFoundError:
    known_face_encoding = []

try:
    with open(known_faces_names_file, "rb") as f:
        known_faces_names = pickle.load(f)
except FileNotFoundError:
    known_faces_names = []

# Function to generate speech
def generate_speech(text):
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
    user_image = face_recognition.load_image_file("F:/Coding/Python_VS_Code/sec_proj/photos/" + name.replace(" ", "_") + ".jpg")
    user_encoding = face_recognition.face_encodings(user_image)[0]
    known_face_encoding.append(user_encoding)
    known_faces_names.append(name)
    with open(known_face_encoding_file, "wb") as f:
        pickle.dump(known_face_encoding, f)
    with open(known_faces_names_file, "wb") as f:
        pickle.dump(known_faces_names, f)
        
# Function to generate speech for an alert
def generate_alert_speech(object_label):
    text = f"Alert! {object_label} detected!"
    tts = gTTS(text=text, lang="en", slow=False)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    pygame.init()
    pygame.mixer.music.load(mp3_fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Modified detect_and_recognize function to detect and classify specified objects
def detect_and_recognize(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Initialize flags for specific object detection
    alert_detected = False
    detected_object_label = ""

    # Perform object detection
    results = model(frame_rgb)

    # Process the detections
    for det in results.pred[0]:
        label = classes[int(det[5])]
        bbox = det[:4].int().cpu().numpy()
        x, y, x_max, y_max = bbox

        # If the detected object is one of the specified classes
        if label in ["bag", "cell phone", "knife", "gun", "laptop", "mouse", "remote", "keyboard"]:
            # Convert to integer as OpenCV expects integers
            x, y, x_max, y_max = int(x), int(y), int(x_max), int(y_max)
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Set the alert flag to True and store the detected object label
            alert_detected = True
            detected_object_label = label

    # Generate an alert if any specified object is detected
    if alert_detected:
        generate_alert_speech(detected_object_label)


    # Recognize faces
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_faces_names[first_match_index]
        else:
            # Beep and generate speech for an unknown person
            winsound.Beep(1000, 500)  # Adjust the frequency and duration of the beep as needed
            generate_speech("Unknown person detected!")

        # Adjust position for displaying the name
        top -= 10
        cv2.putText(frame, f"{name}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Function to detect and recognize people
def detect_and_recognize_people():
    video = cv2.VideoCapture(0)

    while True:
        _, frame = video.read()

        # Detect and recognize persons and faces
        frame_processed = detect_and_recognize(frame)

        # Resize the frame to a larger size
        frame_resized = cv2.resize(frame_processed, (1280, 850))  # Adjust dimensions as needed

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

# Uncomment the following line to add a user (provide the name)
# add_user("Soumyajit")
detect_and_recognize_people()