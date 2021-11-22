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

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam, whattup")

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
    #                    interpolation=cv2.INTER_AREA)
    # Converting BGR image into a RGB image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # load classifier
    face_data = face_detect.detectMultiScale(frame, 1.5, 2)
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
        # face_resized_color = cv2.resize(roi, (96, 96), interpolation = cv2.INTER_AREA)

        # sunglasses = cv2.imread("sunglasses_5.png", cv2.IMREAD_UNCHANGED)
        # sunglass_width = int(frame.shape[0])
        # sunglass_height = int(frame.shape[1])
        # sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
        # transparent_region = sunglass_resized[:,:,:3] != 0
        # face_resized_color[int(frame.shape[1]):int(frame.shape[1])+sunglass_height, int(frame.shape[0]):int(frame.shape[0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
        # # face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
        
        # # Resize the face_resized_color image back to its original shape
        # frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, frame.shape, interpolation = cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # revert back to BGR format

    cv2.imshow('Input', frame)
    # print("sleeping")
    # time.sleep(5)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
