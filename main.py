import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained facial expression recognition model
model = load_model("fer2013_mini_XCEPTION.102-0.66.hdf5")

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define a function to detect facial expressions
def detect_expression(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (64, 64))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))
        result = model.predict(reshaped_face)
        emotion = np.argmax(result)

        # Define the emotions (labels) in the same order as the model output
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        # Draw the detected face and the predicted emotion on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotions[emotion], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_expression = detect_expression(frame)
    cv2.imshow("Facial Expression Detection", frame_with_expression)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
