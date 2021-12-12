import cv2
import time
import imutils
from imutils import face_utils
from random import randint
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
        self.dlib_face_features = models['dlib_face_features']

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
        end = time.time()
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

    def frontalfacedetection(self, modalities):
        imag_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # perform face detection using dlib's face detector
        start = time.time()
        rects = self.dlibfrontalface(imag_rgb, 1) ## 0 upsamples for speed purposes
        end = time.time()
        print("[INFO] dlibfrontalface took {:.4f} seconds".format(end - start))
        boxes = [self.convert_and_trim_bb(self.frame, r) for r in rects]
        for (x, y, w, h) in boxes: # draw the bounding box on our image
	        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return

    def cnn_face_detection(self):
        imag_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # perform face detection using dlib's face detector
        start = time.time()
        rects = self.cnn_face_detection_model_v1(imag_rgb, 0) ## 0 upsamples for speed purposes
        end = time.time()
        print("[INFO] cnn_face_detection_model_v1 took {:.4f} seconds".format(end - start))
        boxes = [self.convert_and_trim_bb(self.frame, r.rect) for r in rects]
        for (x, y, w, h) in boxes:  # draw the bounding box on our image
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

    def rect_to_bb(self, rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # return a tuple of (x, y, w, h)
        return (x, y, w, h)

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def face_features(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        rects = self.dlibfrontalface(gray, 1)

        for (i, rect) in enumerate(rects):
        	# determine the facial landmarks for the face region, then
        	# convert the facial landmark (x, y)-coordinates to a NumPy
        	# array
            colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
            alpha = 0.75
            start = time.time()
            shape = self.dlib_face_features(gray, rect)
            end = time.time()
            print("[INFO] dlib_face_features took {:.4f} seconds".format(end - start))
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            '''
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            cv2.putText(self.frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(self.frame, (x, y), 1, (0, 0, 255), -1)
            '''
            # loop over the face parts individually
            # loop over the facial landmark regions individually
            print("FACIAL_LANDMARKS_IDXS.keys() : \n", FACIAL_LANDMARKS_IDXS.keys())
            for (i, name) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
                # grab the (x, y)-coordinates associated with the
                # face landmark
                (j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
                pts = shape[j:k]
                # check if are supposed to draw the jawline
                if name == "jaw":
                    # print()
                    # since the jawline is a non-enclosed facial region,
                    # just draw lines between the (x, y)-coordinates
                    for l in range(1, len(pts)):
                        ptA = tuple(pts[l - 1])
                        ptB = tuple(pts[l])
                        print("----- ", i, "----")
                        cv2.line(self.frame, ptA, ptB, colors[i], 2)
                # otherwise, compute the convex hull of the facial
                # landmark coordinates points and display it
                else:
                    print("got to hull")
                    hull = cv2.convexHull(pts)
                    cv2.drawContours(self.frame, [hull], -1, colors[i], -1)
                    # cv2.addWeighted(self.frame, alpha, self.frame, 1-alpha, 0, self.frame)
        return
