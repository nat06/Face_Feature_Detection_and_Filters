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

## NOTE: haar-cascades models also available in /Users/nathanpollet/anaconda3/envs/XParis/lib/python3.9/site-packages/cv2/data/


def main():
    cap = cv2.VideoCapture(0)
    arr = []

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam.")

    while True:
        ret, inputframe = cap.read()
        numFaces, outputframe = desiredAction(
            "face_eyes_detection_and_blurring", inputframe)
        arr.append(numFaces)
        cv2.imshow('Input', outputframe)

        c = cv2.waitKey(1)
        if c == 27:
            break

    print(arr)
    cap.release()
    cv2.destroyAllWindows()


def desiredAction(name, frame):
    # Converting BGR image into a RGB image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    haarcascade_path = "/Users/nathanpollet/anaconda3/envs/XParis/share/opencv4/haarcascades/"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if name == "face_eyes_detection_and_blurring":
        ## here, returned variable 'val' is the number of detected (and blurred) faces
        # model location within conda env opencv4 installation
        face_detect = cv2.CascadeClassifier(
            haarcascade_path + 'haarcascade_frontalface_default.xml')  # load classifier
        face_data = face_detect.detectMultiScale(
            frame, scaleFactor=1.25, minNeighbors=6, minSize=(30, 30))
        eye_cascade = cv2.CascadeClassifier(
            haarcascade_path + 'haarcascade_eye_tree_eyeglasses.xml')  # load classifier
        val = len(face_data)
        h, w = frame.shape[:2]
        # computing Kernel width and height for efficient blurring
        kernel_width = (w // 9) | 1
        kernel_height = (h // 9) | 1

        # Draw rectangles and blur detected faces
        for (x, y, w, h) in face_data:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # roi = cv2.GaussianBlur(roi, (kernel_width, kernel_height), 0)

            # impose this blurred image on original image to get final image
            frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

        # revert back to BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return val, frame


def eyeDetection(face):
    ## here, returned variable 'val' is the number of detected pairs of eyes
    # model location within conda env opencv4 installation
    haarcascade_path = "/Users/nathanpollet/anaconda3/envs/XParis/share/opencv4/haarcascades/"

    eye_data = eye_cascade.detectMultiScale(
        frame, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30))
    val = len(face_data)
    return val, face


if __name__ == "__main__":
    main()
