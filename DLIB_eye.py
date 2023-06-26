import cv2

# import the necessary packages
from scipy.spatial import distance as dist
import dlib
import cv2

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# plt.close('all')

##################################################
### Real time data plotter in opencv (python)  ###
## Plot integer data for debugging and analysis ##
## Contributors - Vinay @ www.connect.vin       ##
## For more details, check www.github.com/2vin  ##
##################################################

import cv2
import numpy as np


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# def process_data(i, xs, ys):
while True:

    ret, img = cap.read()
    # print(ret)
    frame = img

    if frame is not None:
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = detector(gray)
        if len(faces) > 0:

            if len(faces) == 1:
                face = faces[0]
            
            elif len(faces) > 1:
                big_area = 0
                big_face = 0
                for i in range(1, len(faces)):
                    # extract the coordinates of the bounding box
                    area = abs((faces[i].right()-faces[i].left())*(faces[i].bottom()-faces[i].top()))
                    if area > big_area:
                        big_area = area
                        big_face = i
                face = faces[big_face]

            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
            shape = predictor(gray, face)

            # face_frame = frame[y1:y2, x1:x2]
            # face_frame = cv2.resize(face_frame, (640, 480))

            right_eye = [[shape.part(36).x, shape.part(36).y],
                        [shape.part(37).x, shape.part(37).y], 
                        [shape.part(38).x, shape.part(38).y],
                        [shape.part(39).x, shape.part(39).y],
                        [shape.part(40).x, shape.part(40).y],
                        [shape.part(41).x, shape.part(41).y]]
            
            left_eye = [[shape.part(42).x, shape.part(42).y],
                        [shape.part(43).x, shape.part(43).y], 
                        [shape.part(44).x, shape.part(44).y],
                        [shape.part(45).x, shape.part(45).y],
                        [shape.part(46).x, shape.part(46).y],
                        [shape.part(47).x, shape.part(47).y]]
            
            delta = int((shape.part(39).x - shape.part(36).x)/3)
            delta_y = int(1.5*delta)
            left_eye_frame = img[(shape.part(40).y-delta_y):(shape.part(37).y+delta_y), (shape.part(36).x-delta):(shape.part(39).x+delta)]

            # cv2.resize(face_frame, (640, 480), interpolation = cv2.INTER_AREA)

            for landmark in right_eye:
                cv2.circle(frame, landmark, 4, (255, 0, 0), -1)

            for landmark in left_eye:
                cv2.circle(frame, landmark, 4, (255, 0, 0), -1)

            EAR = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye))

            data = EAR

            # cv2.imshow("Drowsiness detection", frame)
            try:
                cv2.imshow("Left eye", left_eye_frame)
            except:
                pass
            if cv2.waitKey(1)&0xFF==ord("q"): 
                break
