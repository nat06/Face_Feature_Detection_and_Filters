import cv2
import numpy as np
import dlib
import math
import os
from datetime import datetime
import pickle

os.chdir("/home/laura/Documents/Polytechnique/MScT - M1/INF573 Image Analysis and Computer Vision/INF573 - Final Project/INF573---Project")

def pig_filter(frame, landmarks) :

    # Required landmarks to place an image onto the nose
    nosetop = (landmarks.part(29).x, landmarks.part(29).y)
    nosemid = (landmarks.part(30).x, landmarks.part(30).y)
    noseleft = (landmarks.part(31).x, landmarks.part(31).y)
    noseright = (landmarks.part(35).x, landmarks.part(35).y)

    # Loading filter
    pig_image = cv2.imread("filters/pig_nose-removebg-preview.png")
    og_pig_h, og_pig_w, pig_channels = pig_image.shape

    pig_gray = cv2.cvtColor(pig_image, cv2.COLOR_BGR2GRAY)
    ret, original_mask = cv2.threshold(pig_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)

    pig_width = int(1.5*abs(noseright[0] - noseleft[0]))
    pig_height = int(abs(pig_width * og_pig_h/og_pig_w ))

    up_center = (int(nosemid[0] - pig_width / 2),int(nosemid[1] - pig_height / 2))
    
    pig_area = frame[ up_center[1] : up_center[1] + pig_height, up_center[0] : up_center[0] + pig_width]
    pig_img = cv2.resize(pig_image, (pig_width, pig_height))
    pig_mask = cv2.resize(original_mask, (pig_width, pig_height))
    pig_mask_inv = cv2.resize(original_mask_inv, (pig_width, pig_height))


    roi_bg = cv2.bitwise_and(pig_area, pig_area, mask = pig_mask)
    roi_fg = cv2.bitwise_and(pig_area, pig_area,mask= pig_mask_inv)
    final_frame = cv2.add(roi_bg, pig_img)

    frame[up_center[1] : up_center[1] + pig_height, up_center[0] : up_center[0] + pig_width] = final_frame

    return frame

def piercing_filter(frame, landmarks) :

    # Required landmarks to place an image onto the nose
    nostril_left = (landmarks.part(32).x, landmarks.part(32).y)
    nosemid = (landmarks.part(33).x, landmarks.part(33).y)
    nostril_right = (landmarks.part(34).x, landmarks.part(34).y)
    lip_top = (landmarks.part(51).x, landmarks.part(51).y)

    # Loading filter
    piercing = cv2.imread("filters/nosepiercing-removebg-preview.png")
    og_piercing_h, og_piercing_w, piercing_channels = piercing.shape

    piercing_gray = cv2.cvtColor(piercing, cv2.COLOR_BGR2GRAY)
    ret, original_mask = cv2.threshold(piercing_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)

    piercing_width = int(4*abs(nostril_right[0] - nostril_left[0]))
    # pig_height = int(abs(pig_width * og_pig_h/og_pig_w ))
    piercing_height = int(2*abs(nosemid[1] - lip_top[1]))

    up_center = (int(nosemid[0] - piercing_width / 2),int(nosemid[1] - piercing_height / 2))
    
    piercing_area = frame[ up_center[1] : up_center[1] + piercing_height, up_center[0] : up_center[0] + piercing_width]
    piercing_img = cv2.resize(piercing, (piercing_width, piercing_height))
    piercing_mask = cv2.resize(original_mask, (piercing_width, piercing_height))
    piercing_mask_inv = cv2.resize(original_mask_inv, (piercing_width, piercing_height))


    roi_bg = cv2.bitwise_and(piercing_area, piercing_area, mask = piercing_mask)
    roi_fg = cv2.bitwise_and(piercing_area, piercing_area,mask= piercing_mask_inv)
    final_frame = cv2.add(roi_bg, piercing_img)

    frame[up_center[1] : up_center[1] + piercing_height, up_center[0] : up_center[0] + piercing_width] = final_frame

    return frame

def glasses_filter(frame, landmark) :

     # Required landmarks to place an image around eyes
    foreheadtop_left = (landmarks.part(77).x, landmarks.part(77).y)
    foreheadbottom_left = (landmarks.part(1).x, landmarks.part(1).y)
    foreheadbottom_right = (landmarks.part(15).x, landmarks.part(15).y)

    # Loading filter
    glasses = cv2.imread("filters/sunglasses_5.png")
    og_glasses_h, og_glasses_w, glasses_channels = glasses.shape

    glasses_gray = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)
    ret, original_mask = cv2.threshold(glasses_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)

    glasses_width = int(1.5*abs(foreheadbottom_right[0] - foreheadbottom_left[0]))
    glasses_height = int(abs(glasses_width * og_glasses_h/og_glasses_w))

    up_center = (int(foreheadtop_left[0] - glasses_width / 2),int(foreheadtop_left[1] - glasses_height / 2))
    
    glasses_area = frame[ up_center[1] : up_center[1] + glasses_height, up_center[0] : up_center[0] + glasses_width]
    glasses_img = cv2.resize(glasses, (glasses_width, glasses_height))
    pig_mask = cv2.resize(original_mask, (glasses_width, glasses_height))
    pig_mask_inv = cv2.resize(original_mask_inv, (glasses_width, glasses_height))


    roi_bg = cv2.bitwise_and(glasses_area,glasses_area,mask = pig_mask)
    roi_fg = cv2.bitwise_and(glasses_area,glasses_area,mask= pig_mask_inv)
    final_frame = cv2.add(roi_bg, glasses_img)

    frame[up_center[1] : up_center[1] + glasses_height, up_center[0] : up_center[0] + glasses_width] = final_frame

    return frame


##################################### INTERACTIVE PART ###################################
##########################################################################################


# Loading models
landmark_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

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

    # Pig filter
    if key == ord('p'):

        while cap.isOpened() :
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pig_mask = np.zeros((frame.shape[:2][0], frame.shape[:2][1], 3))
            faces = landmark_detector(frame)
            
            for face in faces:
                landmarks = landmark_predictor(gray_frame, face)
                frame = pig_filter(frame, landmarks)

            cv2.imshow('frame', frame)

            # If q is pressed quit
            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing filter")
                break

    # Glasses filter
    if key == ord('g') :

        while cap.isOpened() :
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pig_mask = np.zeros((frame.shape[:2][0], frame.shape[:2][1], 3))
            faces = landmark_detector(frame)
            
            for face in faces:
                landmarks = landmark_predictor(gray_frame, face)
                frame = glasses_filter(frame, landmarks)

            cv2.imshow('frame', frame)

            # If q is pressed quit
            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing filter")
                break
    
    # Piercing filter
    if key == ord('s') :

        while cap.isOpened() :
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pig_mask = np.zeros((frame.shape[:2][0], frame.shape[:2][1], 3))
            faces = landmark_detector(frame)
            
            for face in faces:
                landmarks = landmark_predictor(gray_frame, face)
                frame = piercing_filter(frame, landmarks)

            cv2.imshow('frame', frame)

            # If q is pressed quit
            if cv2.waitKey(1) & 0xFF == ord(' '):
                print("Removing filter")
                break
    
    if key == ord('q'):
        break

    

cap.release()
cv2.destroyAllWindows()







# Trying to enable a filter with keyboard

################################### TRY1 ###################################
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     pig_mask = np.zeros((img_w, img_h, 3))
    
#     faces = landmark_detector(frame)

#     # If q is pressed quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("q pressed")
#         break

#     if keyboard.is_pressed('p') :
        
#         for face in faces:
#             landmarks = landmark_predictor(gray_frame, face)
#             frame = pig_filter(frame, landmarks)
        
#     cv2.imshow('frame', frame)

################################### TRY2 ###################################
# import tty, sys, termios

# filedescriptors = termios.tcgetattr(sys.stdin)
# tty.setcbreak(sys.stdin)
# x = 0

# while(cap.isOpened()):
    
#     ret, frame = cap.read()
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     pig_mask = np.zeros((img_w, img_h, 3))
    
#     cv2.imshow('frame', frame)
    
#     print("hi!")
#     x=sys.stdin.read(1)[0]
#     print("hhhhhhhhi")

#     while x == "p":
#         print("If condition is met")

#         x=sys.stdin.read(1)[0]
#         print("You pressed", x)

#     print("imout")

# termios.tcsetattr(sys.stdin, termios.TCSADRAIN,filedescriptors)