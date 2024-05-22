import cv2
import openvino

# Load the pre-trained action recognition model
model = openvino.Model('path/to/model.xml')

# Create a video capture object
cap = cv2.VideoCapture(0)

# Loop over the webcam feed
while True:
    # Capture a frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(float) / 255.0

    # Perform inference
    input_tensor = openvino.Tensor(frame)
    output_tensor = model.infer({input_tensor: input_tensor})

    # Get the predicted action label
    output = output_tensor.numpy()
    label_id = output.argmax()
    label = model.classes[label_id]

    # Display the predicted action label on the frame
    cv2.putText(frame, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
