import cv2
import numpy as np
import dlib
import math
import os

os.chdir("/home/laura/Documents/Polytechnique/MScT - M1/INF573 Image Analysis and Computer Vision/INF573 - Final Project/INF573---Project")

def pig_filter(frame, landmarks) :

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

    dog_width = int(1.5*abs(noseright[0] - noseleft[0]))
    dog_height = int(abs(dog_width * og_pig_h/og_pig_w ))

    up_center = (int(nosemid[0] - dog_width / 2),int(nosemid[1] - dog_height / 2))
    down_center = (int(nosemid[0] + dog_width / 2),int(nosemid[1] + dog_height / 2))
    
    dog_area = frame[up_center[1]: up_center[1] + dog_height,up_center[0]: up_center[0] + dog_width]
    dog_img = cv2.resize(pig_image, (dog_width, dog_height))
    pig_mask = cv2.resize(original_mask, (dog_width, dog_height))
    pig_mask_inv = cv2.resize(original_mask_inv, (dog_width, dog_height))


    roi_bg = cv2.bitwise_and(dog_area,dog_area,mask = pig_mask)
    roi_fg = cv2.bitwise_and(dog_area,dog_area,mask= pig_mask_inv)
    final_frame = cv2.add(roi_bg, dog_img)

    frame[up_center[1]: up_center[1] + dog_height,up_center[0]: up_center[0] + dog_width] = final_frame

    return frame

# Loading models
landmark_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]

while(cap.isOpened()):
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pig_mask = np.zeros((img_w, img_h, 3))
    
    faces = landmark_detector(frame)

    for face in faces:
        landmarks = landmark_predictor(gray_frame, face)
        frame = pig_filter(frame, landmarks)
    
    cv2.imshow('frame', frame)

    # If q is pressed quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break

cap.release()

cv2.destroyAllWindows()