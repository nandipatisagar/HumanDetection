import cv2
import numpy as np
# Load HOG classifier
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def human_detection_tracking(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Loop through video frames
    while True:
        # Read a frame
        ret, frame = cap.read()

        # Break if no frame is captured
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 480))

        # Detect humans in the frame
        humans, _ = hog.detectMultiScale(frame, winStride=(8, 8))

        # Draw bounding boxes around detected humans
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Human Detection and Tracking', frame)

        # Wait for key press to exit
        if cv2.waitKey(1) == ord('q'):
            break

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


human_detection_tracking('D:\miniproject\pedestrian-126503.mp4')