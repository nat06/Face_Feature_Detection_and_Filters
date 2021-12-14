import cv2
import numpy as np
import dlib
import math
import os
from frameObject import frameObject
from datetime import datetime



# CHANGE THIS TO YOUR PATH TO INF573--Project
path = "/home/laura/Documents/Polytechnique/MScT - M1/INF573 Image Analysis and Computer Vision/INF573 - Final Project/INF573---Project"
os.chdir(path)

class filterObject(frameObject):

    def __init__(self, inputframe, models, name) :
        self.name = name
        super().__init__(inputframe, models)

    def get_name(self) :
        return self.name

    def get_function(self) :
    
        return FILTERS[self.name]

    ################################# Function to add filters to object #################################

    def pig_filter(self, landmarks) :

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
        
        pig_area = self.frame[ up_center[1] : up_center[1] + pig_height, up_center[0] : up_center[0] + pig_width]
        pig_img = cv2.resize(pig_image, (pig_width, pig_height))
        pig_mask = cv2.resize(original_mask, (pig_width, pig_height))
        pig_mask_inv = cv2.resize(original_mask_inv, (pig_width, pig_height))


        roi_bg = cv2.bitwise_and(pig_area, pig_area, mask = pig_mask)
        roi_fg = cv2.bitwise_and(pig_area, pig_area,mask= pig_mask_inv)
        final_frame = cv2.add(roi_bg, pig_img)

        self.frame[up_center[1] : up_center[1] + pig_height, up_center[0] : up_center[0] + pig_width] = final_frame

        return self.frame

    def piercing_filter(self, landmarks) :

        # Required landmarks to place an image onto the nose
        nostril_left = (landmarks.part(32).x, landmarks.part(32).y)
        nosemid = (landmarks.part(33).x, landmarks.part(33).y)
        nostril_right = (landmarks.part(34).x, landmarks.part(34).y)
        lip_top = (landmarks.part(51).x, landmarks.part(51).y)

        # Loading filter
        piercing = cv2.imread("filters/another_septum-removebg-preview.png")
        og_piercing_h, og_piercing_w, piercing_channels = piercing.shape

        piercing_gray = cv2.cvtColor(piercing, cv2.COLOR_BGR2GRAY)
        ret, original_mask = cv2.threshold(piercing_gray, 10, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        # # For septum 1
        # piercing_width = int(4*abs(nostril_right[0] - nostril_left[0]))
        # piercing_height = int(2*abs(nosemid[1] - lip_top[1]))
        # For septum 2
        piercing_width = int(2*abs(nostril_right[0] - nostril_left[0]))
        piercing_height = int(abs(nosemid[1] - lip_top[1]))

        up_center = (int(nosemid[0] - piercing_width / 2),int(nosemid[1] - piercing_height / 2))

        piercing_area = self.frame[ up_center[1] : up_center[1] + piercing_height, up_center[0] : up_center[0] + piercing_width]

        piercing_img = cv2.resize(piercing, (piercing_width, piercing_height))
        piercing_mask = cv2.resize(original_mask, (piercing_width, piercing_height))
        piercing_mask_inv = cv2.resize(original_mask_inv, (piercing_width, piercing_height))

        roi_bg = cv2.bitwise_and(piercing_area, piercing_area, mask = piercing_mask)
        roi_fg = cv2.bitwise_and(piercing_area, piercing_area,mask= piercing_mask_inv)
        final_frame = cv2.add(roi_bg, piercing_img)

        self.frame[up_center[1] : up_center[1] + piercing_height, up_center[0] : up_center[0] + piercing_width] = final_frame

        return self.frame

    def devil_horns_filter(self, landmarks) :

        # Required landmarks to place an image onto the nose
        hairlinel = (landmarks.part(75).x, landmarks.part(75).y)
        hairliner = (landmarks.part(74).x, landmarks.part(74).y)
        highest_hairline = (landmarks.part(69).x, landmarks.part(69).y)

        # Loading filter
        horns = cv2.imread("filters/neon-devil-horns-removebg-preview.png")
        og_horns_h, og_horns_w, horns_channels = horns.shape

        horns_gray = cv2.cvtColor(horns, cv2.COLOR_BGR2GRAY)
        ret, original_mask = cv2.threshold(horns_gray, 10, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        horns_width = int(1.25*abs(hairliner[0] - hairlinel[0]))
        horns_height = int(0.5*og_horns_h)

        up_center = (int(highest_hairline[0] - horns_width/3), int(highest_hairline[1] - horns_height +50))
        print(up_center)


        horns_area = self.frame[ up_center[1] : up_center[1] + horns_height, up_center[0] : up_center[0] + horns_width]

        piercing_img = cv2.resize(horns, (horns_width, horns_height))
        piercing_mask = cv2.resize(original_mask, (horns_width, horns_height))
        piercing_mask_inv = cv2.resize(original_mask_inv, (horns_width, horns_height))


        roi_bg = cv2.bitwise_and(horns_area, horns_area, mask = piercing_mask)
        roi_fg = cv2.bitwise_and(horns_area, horns_area,mask= piercing_mask_inv)
        final_frame = cv2.add(roi_bg, piercing_img)

        self.frame[up_center[1] : up_center[1] + horns_height, up_center[0] : up_center[0] + horns_width] = final_frame

        return self.frame

    def diadelosmuertos_filter(self, landmarks) :

        # Required landmarks to place an image onto the face
        temple_left = (landmarks.part(0).x, landmarks.part(0).y)
        temple_right = (landmarks.part(16).x, landmarks.part(16).y)
        middle_hairline = (landmarks.part(71).x, landmarks.part(71).y)
        middle_jaw = (landmarks.part(8).x, landmarks.part(8).y)

        # Loading filter
        paint = cv2.imread("filters/tats-removebg-preview.png")
        # paint = cv2.imread('filters/snapchat-psd-snapchat-dog-effect.png')
        og_paint_h, og_paint_w, channels = paint.shape
        # paint = cv2.bitwise_not(paint)
        print(paint)

        paint_gray = cv2.cvtColor(paint, cv2.COLOR_BGR2GRAY)
        ret, original_mask = cv2.threshold(paint_gray, 10, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        paint_width = int(abs(temple_right[0] - temple_left[0]))
        paint_height = int(abs(middle_hairline[1]- middle_jaw[1]))
        print(paint_width)
        print(paint_height)

        up_center = (int(middle_jaw[0] - 0.45*paint_width), int(middle_jaw[1] - paint_height))
        print(up_center)


        paint_area = self.frame[ up_center[1] : up_center[1] + paint_height, up_center[0] : up_center[0] + paint_width]
        print(paint_area.shape)
        paint_img = cv2.resize(paint, (paint_width, paint_height))
        paint_mask = cv2.resize(original_mask, (paint_width, paint_height))
        paint_mask = cv2.resize(original_mask_inv, (paint_width, paint_height))
        print(paint_mask.shape)

        roi_bg = cv2.bitwise_and(paint_area, paint_area, mask = paint_mask)
        roi_fg = cv2.bitwise_and(paint_area, paint_area,mask= paint_mask)
        print(paint_img.shape)
        print(roi_bg.shape)
        final_frame = cv2.add(roi_fg, paint_img)

        self.frame[up_center[1] : up_center[1] + paint_height, up_center[0] : up_center[0] + paint_width] = final_frame

        return self.frame

    def tears_filter(self, landmarks) :

        # Required landmarks to place an image around both eyes
        ##  Left eye
        eye_ll = (landmarks.part(36).x, landmarks.part(36).y)  # Left eye Left side
        eye_llm = (landmarks.part(41).x, landmarks.part(41).y) # Left eye Left Middle
        eye_lrm = (landmarks.part(40).x, landmarks.part(40).y) # Left eye Right Middle
        eye_lr = (landmarks.part(39).x, landmarks.part(39).y)  # Left eye Right side
        ## Right eye
        eye_rl = (landmarks.part(42).x, landmarks.part(42).y)  # Right eye Left side
        eye_rlm = (landmarks.part(47).x, landmarks.part(47).y) # Right eye Left Middle
        eye_rrm = (landmarks.part(46).x, landmarks.part(46).y) # Right eye Right Middle
        eye_rr = (landmarks.part(45).x, landmarks.part(45).y)  # Right eye Right side
        ## Mid nose
        nosemid = (landmarks.part(33).x, landmarks.part(33).y)
        ## Between eyebrows
        btw_eyebrows = (landmarks.part(27).x, landmarks.part(27).y)
        bridge = (landmarks.part(28).x, landmarks.part(28).y)

        # Loading filter
        tears = cv2.imread("filters/tears-removebg-preview.png")

        tears_gray = cv2.cvtColor(tears, cv2.COLOR_BGR2GRAY)
        ret, original_mask = cv2.threshold(tears_gray, 10, 255, cv2.THRESH_BINARY_INV)
        original_mask_inv = cv2.bitwise_not(original_mask)

        tears_width = int(2*abs(eye_rr[0] - eye_ll[0]))
        tears_height = int(2*abs(eye_ll[1] - nosemid[1]))

        # up_center = (int(btw_eyebrows[0] - tears_width / 2), int(btw_eyebrows[1] - tears_height / 2))
        up_center = (int(btw_eyebrows[0] - tears_width / 2), int(btw_eyebrows[1] - tears_height/6 + 20))

        tears_area = self.frame[ up_center[1] : up_center[1] + tears_height, up_center[0] : up_center[0] + tears_width]

        tears_img = cv2.resize(tears, (tears_width, tears_height))
        tears_mask = cv2.resize(original_mask, (tears_width, tears_height))
        tears_mask_inv = cv2.resize(original_mask_inv, (tears_width, tears_height))

        roi_bg = cv2.bitwise_and(tears_area, tears_area, mask = tears_mask)
        roi_fg = cv2.bitwise_and(tears_area, tears_area,mask= tears_mask_inv)
        final_frame = cv2.add(roi_bg, tears_img)

        self.frame[up_center[1] : up_center[1] + tears_height, up_center[0] : up_center[0] + tears_width] = final_frame

        return self.frame

    def glasses_filter(self, landmarks) :

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
        
        glasses_area = self.frame[ up_center[1] : up_center[1] + glasses_height, up_center[0] : up_center[0] + glasses_width]
        glasses_img = cv2.resize(glasses, (glasses_width, glasses_height))
        pig_mask = cv2.resize(original_mask, (glasses_width, glasses_height))
        pig_mask_inv = cv2.resize(original_mask_inv, (glasses_width, glasses_height))


        roi_bg = cv2.bitwise_and(glasses_area,glasses_area,mask = pig_mask)
        roi_fg = cv2.bitwise_and(glasses_area,glasses_area,mask= pig_mask_inv)
        final_frame = cv2.add(roi_bg, glasses_img)

        self.frame[up_center[1] : up_center[1] + glasses_height, up_center[0] : up_center[0] + glasses_width] = final_frame

        return self.frame

FILTERS = {'p': filterObject.pig_filter, 's' : filterObject.piercing_filter, 'd' : filterObject.devil_horns_filter, 
           't' : filterObject.tears_filter, 'm' : filterObject.diadelosmuertos_filter, 'g' : filterObject.glasses_filter}


if __name__ == "__main__" :

    from retinaface import RetinaFace

    models = {}
    models['facedetector_haarcascades'] = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml') # load classifier
    models['eyedetector_haarcascades'] = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_eye_tree_eyeglasses.xml')  # load classifier 
    models['retinaface'] = RetinaFace
    models['dlibfrontalface'] = dlib.get_frontal_face_detector()
    models['cnn_face_detection_model_v1'] = dlib.cnn_face_detection_model_v1("pretrained/mmod_human_face_detector.dat")
    models['dlib_face_features'] = dlib.shape_predictor('pretrained/shape_predictor_81_face_landmarks.dat')

    cap = cv2.VideoCapture(0)
    # If no key is pressed, continue displaying the unedited frame 
    ret, inputframe = cap.read()
    frame = filterObject(inputframe=inputframe, models=models, name='p')
    print(frame.get_function())

    cap.release()
    cv2.destroyAllWindows()
