import cv2 
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import dlib
from retinaface import RetinaFace
from frameObject import frameObject
from filterObject import filterObject
from imutils import face_utils

print(sys.version)
print("cv.__version__ : ", cv2.__version__)
print("Python version : ", sys.version)
print("Libraries imported.")

# CHANGE THIS TO YOUR PATH TO INF573--Project
# path = "/home/laura/Documents/Polytechnique/MScT - M1/INF573 Image Analysis and Computer Vision/INF573 - Final Project/INF573---Project"
# os.chdir(path)

########################################## Preparing models ##########################################

arr = []
models = {}
models['facedetector_haarcascades'] = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml') # load classifier
models['eyedetector_haarcascades'] = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_eye_tree_eyeglasses.xml')  # load classifier 
models['retinaface'] = RetinaFace
models['dlibfrontalface'] = dlib.get_frontal_face_detector()
models['cnn_face_detection_model_v1'] = dlib.cnn_face_detection_model_v1("pretrained/mmod_human_face_detector.dat")
models['dlib_face_features'] = dlib.shape_predictor('pretrained/shape_predictor_81_face_landmarks.dat')
print("done generating models")

#######################################################################################################


cap = cv2.VideoCapture(0)

def general_filter_func(name) :
    while cap.isOpened() :
        ret, inputframe = cap.read()
        frame = filterObject(inputframe=inputframe, models=models, name=name)
        filter_function = frame.get_function()

        g_frame = cv2.cvtColor(frame.getframe(), cv2.COLOR_BGR2GRAY)
        g_frame = frameObject(inputframe=g_frame, models=models)

        # Detects where all the faces are
        faces = frame.get_dlibfrontalface()(frame.getframe())

        for face in faces:

            landmarks = g_frame.get_dlib_face_features()(g_frame.getframe(), face)
            final_frame = filter_function(frame, landmarks)

        cv2.imshow('Frame', final_frame)

        # If q is pressed quit
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print("Removing filter")
            return

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam.")

while cap.isOpened():

    # If no key is pressed, continue displaying the unedited frame 
    ret, inputframe = cap.read()
    frame = frameObject(inputframe, models)
    outputframe = frame.getframe()
    cv2.imshow('Frame', outputframe)

    key = cv2.waitKey(1) & 0xFF

    # Contour points
    if key == ord('c'):

        while(cap.isOpened()):
            ret, inputframe = cap.read()
            frame = frameObject(inputframe, models)

            frame.display_contour()
            
            outputframe = frame.getframe()
            cv2.imshow('Frame', outputframe)
            
            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing feature points")
                break

    # Blur face
    if key == ord('b'):

        while(cap.isOpened()):
            ret, inputframe = cap.read()
            frame = frameObject(inputframe, models)

            frame.faceAndFeaturesDetection("blur")

            outputframe = frame.getframe()
            cv2.imshow('Frame', outputframe)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing feature points")
                break

    # Eye detection
    if key == ord('e'):

        while(cap.isOpened()):
            ret, inputframe = cap.read()
            frame = frameObject(inputframe, models)

            frame.faceAndFeaturesDetection("eyesdetection")

            outputframe = frame.getframe()
            cv2.imshow('Frame', outputframe)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing feature points")
                break

    # Coloured in face (filled in features)
    if key == ord('f'):

        while(cap.isOpened()):

            # Reading in and creating frameObject
            ret, inputframe = cap.read()
            frame = frameObject(inputframe, models)

            # Drawing on the face features
            frame.face_features()
        
            outputframe = frame.getframe()
            cv2.imshow('Frame', outputframe)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

    # Pig filter
    if key == ord('p'):
        general_filter_func('p')

    # Piercing filter
    if key == ord('s') :
        general_filter_func('s')

    # # Devil filter
    if key == ord('d') :
        general_filter_func('d')

    # Dia de los muertos :-(
    if key == ord('m') :
        general_filter_func('m')

    # Tears filter
    if key == ord('t') :
        general_filter_func('t')

    # Glasses filter
    if key == ord('g') :
        general_filter_func('g')

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

