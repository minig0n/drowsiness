import os
import numpy as np
import cv2
import time
import cv2
import sys


#### TRACKER ####---------------------

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

# Inicialização
directory = os.path.dirname(__file__)

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    exit()

# Define an initial bounding box
bbox = (287, 23, 86, 320)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
        
    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break




# Pesos da Rede
weights = os.path.join(directory, "yunet_n_320_320.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

teste = 0
t0 = 0
t1 = 0
delta = 0

while True:
    result, image = capture.read()
    
    # image = cv2.resize(image, (320, 320))
    if result is False:
        cv2.waitKey(0)
        break
    
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    height, width, _ = image.shape
    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(image)
    faces = faces if faces is not None else []

    valid = False
    for face in faces:
        box = list(map(int, face[:4]))
        # print(box)
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
        valid = True

        landmarks = list(map(int, face[4:len(face)-1]))
        landmarks = np.array_split(landmarks, len(landmarks) / 2)
        for landmark in landmarks:
            radius = 5
            thickness = -1
            cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)
            
        confidence = face[-1]
        confidence = "{:.2f}".format(confidence)
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

    if valid:
        teste += 1
        if teste > 5:
            t1 = time.time()
            delta = (t1-t0)/teste
            t0 = time.time()
            teste = 0

    if delta != 0 and delta > 0.001:
        cv2.putText(image, f"FPS: {round(1/delta)}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
    cv2.imshow("face detection", image)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
