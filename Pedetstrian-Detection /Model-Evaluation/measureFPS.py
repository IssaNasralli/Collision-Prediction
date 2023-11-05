import cv2
import numpy as np
import tensorflow as tf
import time
cap = cv2.VideoCapture(0)
# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open the webcam.")
    exit()
# Set desired frame dimensions
total_inference_time = 0
frame_count=0
while True:
    start_time = time.time()
    # Read frame from webcam
    ret, frame = cap.read()
    # Check if the frame is read successfully
    if not ret:
        print("Failed to read frame from the webcam.")
        break
    frame_count=frame_count+1
    average_inference_time = total_inference_time / frame_count
    print("Average Handling Frame Time Without Inferance:", average_inference_time)
    print("Total Time:", total_inference_time)
    print("Frame Count:", frame_count)
    if cv2.waitKey(1) == ord('q'):
        break
# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
