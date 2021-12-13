import cv2
import sys
import os
import time
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from frameObject import frameObject
import dlib
print(sys.version)
print("cv.__version__ : ", cv2.__version__)
print("Python version : ", sys.version)
print("Libraries imported.")

## NOTE: haar-cascades models also available in /Users/nathanpollet/anaconda3/envs/XParis/lib/python3.9/site-packages/cv2/data/

def main():
    cap = cv2.VideoCapture(0)
    arr = []
    haar_path = "/Users/nathanpollet/anaconda3/envs/XParis/share/opencv4/haarcascades/"
    models = {}
    models['facedetector_haarcascades'] = cv2.CascadeClassifier(haar_path + 'haarcascade_frontalface_default.xml') # load classifier
    models['eyedetector_haarcascades'] = cv2.CascadeClassifier(haar_path + 'haarcascade_eye_tree_eyeglasses.xml')  # load classifier 
    models['retinaface'] = RetinaFace
    models['dlibfrontalface'] = dlib.get_frontal_face_detector()
    models['cnn_face_detection_model_v1'] = dlib.cnn_face_detection_model_v1('pretrained/mmod_human_face_detector.dat')
    models['dlib_face_features'] = dlib.shape_predictor('pretrained/shape_predictor_68_face_landmarks.dat')
    print("done loading models")

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam.")
    counter = 0
    while True:
        counter+=1
        ret, inputframe = cap.read()
        frame = frameObject(inputframe, models)
        # frame.faceAndFeaturesDetection("eyesdetection")
        # frame.retinaFacefunc()
        # frame.frontalfacedetection("nadafornow")
        frame.face_features()
        outputframe = frame.getframe()
        cv2.imshow('Input', outputframe)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
