import cv2 
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from frameObject import frameObject
print(sys.version)
print("cv.__version__ : ", cv2.__version__)
print("Python version : ", sys.version)
print("Libraries imported.")

def main():
    cap = cv2.VideoCapture(0)
    arr = []

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam.")

    while True:
        ret, inputframe = cap.read()
        frame = frameObject(inputframe)
        frame.faceAndFeaturesDetection("eyesdetection")
        outputframe = frame.getframe()

        cv2.imshow('Input', outputframe)

        c = cv2.waitKey(1)
        if c == 27:
            break

    print(arr)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
