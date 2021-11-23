import cv2

class frameObject:
    
    # default constructor
    def __init__(self, inputframe):
        self.frame = inputframe
        self.roi = None
        self.numFaces = 0; self.numEyes = 0
        self.h = 0; self.w = 0
        self.eh = 0; self.ew = 0

    ## functions modifying self.frame ##
    def faceAndFeaturesDetection(self, modalities):
    ## returns number of detected faces, modifies self.frame ##
        face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # load classifier
        face_data = face_detect.detectMultiScale(self.frame, scaleFactor=1.15, minNeighbors=7, minSize=(30,30))
        self.numFaces = len(face_data)
        h, w = self.frame.shape[:2]
        self.h = h
        self.w = w
        # computing Kernel width and height for efficient blurring 
        kernel_width = (w // 9) | 1
        kernel_height = (h // 9) | 1
        for (x, y, w, h) in face_data:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.roi = self.frame[y:y+h, x:x+w]
            if "eyesdetection" in modalities:
                self.numEyes += self.eyesdetection()
                self.frame[y:y+self.roi.shape[0], x:x+self.roi.shape[1]] = self.roi
            elif "blur" in modalities:
                self.roi = cv2.GaussianBlur(self.roi, (kernel_width, kernel_height), 0)
                self.frame[y:y+self.roi.shape[0], x:x+self.roi.shape[1]] = self.roi
        
    def eyesdetection(self):
    ## returns number of detected eyes in a face, modifies self.roi ##
        roi_gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')  # load classifier
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(self.roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            self.ew = ew
            self.eh = eh

        return len(eyes)
        
    def getframe(self):
        return self.frame

    def getEyeW(self):
        return self.ew

    def getEyeH(self):
        return self.eh