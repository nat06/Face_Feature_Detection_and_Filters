import cv2
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
print(sys.version)
print("cv.__version__ : ", cv2.__version__)
print("Python version : ", sys.version)
print("Libraries imported.")

cap = cv2.VideoCapture(0)

# model location within conda env opencv4 installation
haarcascade_path = "/Users/nathanpollet/anaconda3/envs/XParis/share/opencv4/haarcascades/"

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam, whattup")

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
    #                    interpolation=cv2.INTER_AREA)
    # Converting BGR image into a RGB image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detect = cv2.CascadeClassifier(haarcascade_path + 'haarcascade_frontalface_default.xml') # load classifier
    face_data = face_detect.detectMultiScale(frame, 1.15, 3)
    h, w = frame.shape[:2]
    # computing Kernel width and height for efficient blurring
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1

    # Draw rectangles and blur detected faces
    for (x, y, w, h) in face_data:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (kernel_width, kernel_height), 0)
        # impose this blurred image on original image to get final image
        frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # revert back to BGR format

    cv2.imshow('Input', frame)
    # print("sleeping")
    # time.sleep(5)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
