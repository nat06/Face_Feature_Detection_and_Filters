import cv2 
import sys
import os
import time
import numpy as np
import time
# import tensorflow as tf
# import keras
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from frameObject import frameObject
import dlib
import imutils
from imutils import face_utils
print(sys.version)
print("cv.__version__ : ", cv2.__version__)
print("Python version : ", sys.version)
print("Libraries imported.")

def main():
    cap = cv2.VideoCapture(0)
    arr = []
    models = {}
    models['facedetector_haarcascades'] = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml') # load classifier
    models['eyedetector_haarcascades'] = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_eye_tree_eyeglasses.xml')  # load classifier 
    models['retinaface'] = RetinaFace
    models['dlibfrontalface'] = dlib.get_frontal_face_detector()
    models['cnn_face_detection_model_v1'] = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
    models['dlib_face_features'] = dlib.shape_predictor('models/shape_predictor_81_face_landmarks.dat')
    print("done generating models")


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
        # This allows you to get the coordingate of the mouth, inner_mouth, left and right eyebrow, left and right eye, nose, jaw
        frame.face_features()
        outputframe = frame.getframe()
        # numFaces, outputframe = desiredAction("face_eyes_detection_and_blurring", inputframe)
        # arr.append(numFaces)
        # cv2.imwrite('./heyo.png', outputframe)
        cv2.imshow('Input', outputframe)

        # if counter == 10:
        #     break

        c = cv2.waitKey(1)
        if c == 27:
            break

    # print(arr)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
