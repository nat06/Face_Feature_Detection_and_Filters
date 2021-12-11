import cv2
import time
class frameObject:
    # default constructor
    def __init__(self, inputframe, models):
        start = time.time()

        self.frame = inputframe
        self.roi = None
        self.numFaces = 0; self.numEyes = 0
        self.facedetector_haarcascades = models['facedetector_haarcascades']
        self.eyedetector_haarcascades = models['eyedetector_haarcascades']
        self.retinaface = models['retinaface']
        self.dlibfrontalface = models['dlibfrontalface']
        self.cnn_face_detection_model_v1 = models['cnn_face_detection_model_v1']

        end = time.time()
        print("[INFO] init took {:.4f} seconds".format(end - start))

## functions modifying self.frame ##
    def faceAndFeaturesDetection(self, modalities):
## returns number of detected faces, modifies self.frame ##
        start = time.time()
        face_data = self.facedetector_haarcascades.detectMultiScale(self.frame, scaleFactor=1.15, minNeighbors=7, minSize=(30,30))
        end = time.time()
        print("[INFO] facedetector_haarcascades took {:.4f} seconds".format(end - start))
        self.numFaces = len(face_data)
        h, w = self.frame.shape[:2]
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
        start = time.time()
        eyes = self.eyedetector_haarcascades.detectMultiScale(roi_gray)
        end = time.time()
        print("[INFO] eyedetector_haarcascades took {:.4f} seconds".format(end - start))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(self.roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        return len(eyes)
        
    def getframe(self):
        return self.frame

    def retinaFacefunc(self):
        print("beginning retinaFacefunc")
        start = time.time()
        resp = self.retinaface.detect_faces(self.frame)
        stop = time.time()
        print("[INFO] retinaface took {:.4f} seconds".format(end - start))
        # num_faces = len(resp.keys())
        h, w = self.frame.shape[:2]
        # computing Kernel width and height for efficient blurring
        kernel_width = (w // 9) | 1
        kernel_height = (h // 9) | 1
        for face in resp.keys():
            coordinates = resp[face]['facial_area']
            x = coordinates[0]; y = coordinates[1]
            x2 = coordinates[2]; y2 = coordinates[3]
            cv2.rectangle(self.frame, (x, y), (x2, y2), (0, 255, 0), 2)
            # roi = self.image[y:y2, x:x2]
            # roi = cv2.GaussianBlur(roi, (kernel_width, kernel_height), 0)
            # impose this blurred image on original image to get final image
            # self.frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
        return

    def dlibfunc(self):
        imag_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # perform face detection using dlib's face detector
        start = time.time()
        # rects = self.dlibfrontalface(imag_rgb, 0) ## 0 upsamples for speed purposes
        rects = self.cnn_face_detection_model_v1(imag_rgb, 0) ## 0 upsamples for speed purposes
        end = time.time()
        # print("[INFO] dlibfrontalface took {:.4f} seconds".format(end - start))
        print("[INFO] cnn_face_detection_model_v1 took {:.4f} seconds".format(end - start))
        boxes = [self.convert_and_trim_bb(self.frame, r) for r in rects]
        for (x, y, w, h) in boxes: # draw the bounding box on our image
	        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return

    def convert_and_trim_bb(self, image, rect):
        # extract the starting and ending (x, y)-coordinates of the
        # bounding box
        startX = rect.left()
        startY = rect.top()
        endX = rect.right()
        endY = rect.bottom()
        # ensure the bounding box coordinates fall within the spatial
        # dimensions of the image
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(endX, image.shape[1])
        endY = min(endY, image.shape[0])
        # compute the width and height of the bounding box
        w = endX - startX
        h = endY - startY
        # return our bounding box coordinates
        return (startX, startY, w, h)
