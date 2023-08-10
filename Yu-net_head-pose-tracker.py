import numpy as np
import cv2
from math import cos, sin
import onnxruntime
import os
import time

VIDEO = True
VIDEO_PATH = 'videos/video1.mp4'

TRACKER_FRAMES = 40  # zero if no tracker is used

TRACKER_ID = 5
# 0 - 'BOOSTING'   - 15 FPS
# 1 - 'MIL'        - 17 FPS
# 2 - 'KCF'        - 40 FPS
# 3 - 'TLD'        - Inutilizável
# 4 - 'MEDIANFLOW' - 45 FPS (menos preciso)
# 5 - 'MOSSE'      - 45 FPS (TOP !!!)

MEAN_FPS_VALUE = 20 


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img

def head_pose(frame, bbox):
    x_min = bbox[0]
    x_max = x_min + bbox[2]
    y_min = bbox[1]
    y_max = y_min + bbox[3]

    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(frame.shape[0], y_max + abs(y_min - y_max) / 10)
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(frame.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, frame.shape[1])

    croped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

    croped_resized_frame = cv2.resize(croped_frame, (224, 224))
    
    # bgr --> rgb
    rgb = croped_resized_frame[..., ::-1]
    # hwc --> chw
    chw = rgb.transpose(2, 0, 1)
    # chw --> nchw
    nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)

    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    outputs = whenet.run(
        output_names = whenet_output_names,
        input_feed = {whenet_input_name: nchw}
    )
    yaw = outputs[0][0][0]
    roll = outputs[0][0][1]
    pitch = outputs[0][0][2]

    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])

    tdx = (x_min+x_max)/2
    tdy = (y_min+y_max)/2
    size = abs(x_max-x_min)//2
    
    return yaw, pitch, roll, tdx, tdy, size


### TRACKER ###-----------

def create_tracker(id):
    
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'MOSSE']
    tracker_type = tracker_types[id]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()

    return tracker

create_tracker(TRACKER_ID)


### YU-Net ###-----------

# Pesos da Rede
directory = ""
weights = os.path.join(directory, "yunet_n_320_320.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))


### WHENet ###-----------

whenet_input_name = None
whenet_output_names = None
whenet_output_shapes = None

whenet = onnxruntime.InferenceSession(
    f'whenet_1x3x224x224_prepost.onnx',
    providers=[
        # 'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
)
whenet_input_name = whenet.get_inputs()[0].name
whenet_output_names = [output.name for output in whenet.get_outputs()]


### VIDEO ###------------

if not VIDEO:
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
else:
    print("[INFO] starting video...")
    vs = cv2.VideoCapture(VIDEO_PATH)

time.sleep(1)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# videoWriter = cv2.VideoWriter('video.avi', fourcc, 20.0, (320, 260))


rate = 0
t0 = 0
t1 = 0
delta = 0
count = 0

box = []

while True:
    ret, img = vs.read()
    frame = img

    if not ret:
        break

    if VIDEO:
        frame = frame[:, frame.shape[0]:]

    frame = cv2.resize(frame, (320, 320))

    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))

    # FACE DETECTION (if count == 0)
    faces = []
    if count == 0:
        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []

        valid = False
        if len(faces) > 0:
            valid = True

            face = faces[0]
                
            box = list(map(int, face[:4]))
            print(f"face detected: {box}")
            
            confidence = "{:.2f}".format(face[-1])
            # landmarks = list(map(int, face[4:len(face)-1]))
            # landmarks = np.array_split(landmarks, len(landmarks) / 2)

            # INIT TRACKER
            tracker = create_tracker(TRACKER_ID)
            ok = tracker.init(frame, box)
            # print(ok)
            # tracker.start_track(frame, box)

            count += 1
    
    # FACE TRACKING
    else:
        old_box = box
        ok, box = tracker.update(frame)
        box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

        if abs(box[0] - old_box[0]) > 30:
            # print(f"tracking failed - bbox: {box}")
            box = old_box

        # print(f"tracking: {old_box} -> {box}")
        count += 1

    if count > TRACKER_FRAMES:
        count = 0


    ### WHENet head pose
    if len(box) > 0 and valid:

        yaw, pitch, roll, tdx, tdy, size = head_pose(frame, box)
        orientation = f'yaw: {int(yaw)}, pitch: {int(pitch)}, roll: {int(roll)}'
        # print(orientation)
        draw_axis(frame, yaw, pitch, roll, tdx, tdy, size)

        ### Draw Face
        cv2.rectangle(frame, box, (0, 0, 255), 2, cv2.LINE_AA)
        # for landmark in landmarks:
        #     cv2.circle(frame, landmark, 2, (0, 0, 255), -1, cv2.LINE_AA)
        position1 = (box[0], box[1] - 10)
        cv2.putText(frame, confidence, position1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, orientation, (10, frame.shape[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    if valid:
        rate += 1
        if rate > MEAN_FPS_VALUE:
            t1 = time.time()
            delta = (t1-t0)/rate
            t0 = time.time()
            rate = 0

    if delta != 0 and delta > 0.001:
        cv2.putText(frame, f"FPS: {round(1/delta)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('test', frame)
    # videoWriter.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

