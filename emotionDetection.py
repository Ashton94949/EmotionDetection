import cv2
import numpy as np
import tensorflow as tf
from collections import deque #For moving average
# Load the trained model
model = tf.keras.models.load_model("emotion_model.keras")  # Ensure this is the correct file

# Load OpenCV's face detector
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")


# Define emotion labels (order should match training labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_history = deque(maxlen=5)

print(" Emotion Detection Started")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to blob format for DNN
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))


    #detect faces useing deep learning
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Only process faces with confidence > 50%
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            # Ensure coordinates are within frame bounds
            x, y, x2, y2 = max(0, x), max(0, y), min(w, x2), min(h, y2)

            # Extract the face safely
            face = frame[y:y2, x:x2]

            # Check if the face is actually detected (not an empty crop)
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue  # Skip if face is too small or invalid
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            face = cv2.equalizeHist(face)

            face = cv2.resize(face, (48, 48))
            face = face / 255.0

            # Reshape for model input
            face = np.expand_dims(face, axis=0)  # Add batch dimension
            face = np.expand_dims(face, axis=-1)  # Add channel dimension

            # Predict emotion
            predictions = model.predict(face)
            emotion_index = np.argmax(predictions)
            emotion_confidence = predictions[0][emotion_index]

            if emotion_confidence > 0.30:
                detected_emotion = emotion_labels[emotion_index]
            else:
                detected_emotion = "Uncertain"

            emotion_history.append(detected_emotion)
            most_common_emotion = max(set(emotion_history), key=emotion_history.count)

            # Draw rectangle around the face and display emotion label
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, most_common_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the video feed with emotion detection
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
