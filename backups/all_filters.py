import cv2
import numpy as np
import dlib
import math
import os
from datetime import datetime
from imutils import face_utils
import pickle
import filter_functions as ff



# CHANGE THIS TO YOUR PATH TO INF573--Project
path = "/home/laura/Documents/Polytechnique/MScT - M1/INF573 Image Analysis and Computer Vision/INF573 - Final Project/INF573---Project"
os.chdir(path)

# Loading models
landmark_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("models/shape_predictor_81_face_landmarks.dat")

FILTERS = {'p': ff.pig_filter, 's' : ff.piercing_filter, 'd' : ff.devil_horns_filter, 
           't' : ff.tears_filter, 'm' : ff.diadelosmuertos_filter, 'g' : ff.glasses_filter}

def get_filters(filter_function) :
    while cap.isOpened() :
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = landmark_detector(frame)
            
            for face in faces:
                landmarks = landmark_predictor(gray_frame, face)
                frame = filter_function(frame, landmarks)

            cv2.imshow('frame', frame)

            # If q is pressed quit
            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing filter")
                return


##################################### INTERACTIVE PART ###################################
##########################################################################################


cap = cv2.VideoCapture(0)
while cap.isOpened():
    
    # If no key is pressed, continue displaying the unedited frame 
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # Contour points
    if key == ord('c'):

        while(cap.isOpened()):
            ret, frame = cap.read()
            dets = landmark_detector(frame, 0)
            for k, d in enumerate(dets):
                shape = landmark_predictor(frame, d)
                landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
                for num in range(shape.num_parts):
                    cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), 1)
            
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing feature points")
                break
    
    # Rectangle around face
    if key == ord('r'):
        while(cap.isOpened()):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = landmark_detector(gray, 1)

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                (168, 100, 168), (158, 163, 32),
                (163, 38, 32), (180, 42, 220)]
                alpha = 0.75
                shape = landmark_predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                for (i, name) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
                    # grab the (x, y)-coordinates associated with the
                    # face landmark
                    (j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
                    pts = shape[j:k]
                    # check if are supposed to draw the jawline
                    if name == "jaw":
                        # since the jawline is a non-enclosed facial region,
                        # just draw lines between the (x, y)-coordinates
                        for l in range(1, len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, colors[i-1], 2)
                    # otherwise, compute the convex hull of the facial
                    # landmark coordinates points and display it
                    else:
                        hull = cv2.convexHull(pts)
                        cv2.drawContours(frame, [hull], -1, colors[i-1], -1)
                        # cv2.addWeighted(frame, alpha, frame, 1-alpha, 0, frame)
                
                # Drawing the additional 13 points for the forehead
                ## Irregularly plotted so have to do it manually
                pts = [shape[77], shape[75], shape[76], shape[68], shape[69], shape[70], shape[71],
                       shape[80], shape[72], shape[73], shape[79], shape[74], shape[78]]
                for l in range(1, len(pts)):
                            ptA = tuple(pts[l - 1])
                            ptB = tuple(pts[l])
                            cv2.line(frame, ptA, ptB, colors[0], 2)
            
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break


    # Pig filter
    if key == ord('p'):

        filter_function = FILTERS['p']
        get_filters(filter_function)
    
    # Piercing filter
    if key == ord('s') :

        filter_function = FILTERS['s']
        get_filters(filter_function)
    
    # Devil filter
    if key == ord('d') :

        filter_function = FILTERS['d']
        get_filters(filter_function)
    
    # Dia de los muertos :
    if key == ord('m') :

        filter_function = FILTERS['m']
        get_filters(filter_function)

    # Tears filter
    if key == ord('t') :

        filter_function = FILTERS['t']
        get_filters(filter_function)

    # Glasses filter
    if key == ord('g') :

        filter_function = FILTERS['g']
        get_filters(filter_function)
    
    # Quit
    if key == ord('q'):
        break

    

cap.release()
cv2.destroyAllWindows()
