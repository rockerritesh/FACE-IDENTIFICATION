import cv2
import numpy as np
import face_recognition
import os
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')




# For video streams:
video_capture = cv2.VideoCapture(0)  # Use the appropriate video source (0 for webcam)

while True:
    ret, frame = video_capture.read()

    # Convert the video frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face region for emotion detection
        face_roi = gray[y:y + h, x:x + w]

        # Detect smiles in the face region
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)

        # Determine the emotion based on the presence of a smile
        if len(smiles) > 0:
            emotion = "Happy"
        else:
            emotion = "Neutral"

        # Display the emotion text near the face rectangle
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()

