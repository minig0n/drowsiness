import numpy as np
import cv2
from math import cos, sin
import onnxruntime
import os
import time


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
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
)
whenet_input_name = whenet.get_inputs()[0].name
whenet_output_names = [output.name for output in whenet.get_outputs()]


### VIDEO ###------------

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# videoWriter = cv2.VideoWriter('video.avi', fourcc, 20.0, (320, 260))


teste = 0
t0 = 0
t1 = 0
delta = 0
while True:
    ret, img = vs.read()
    frame = cv2.resize(img, (320, 260))
    # frame = img

    if not ret:
        continue

    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)
    faces = faces if faces is not None else []

    valid = False
    for face in faces:
        valid = True
        
        # Get detection info
        box = list(map(int, face[:4]))
        confidence = face[-1]
        confidence = "{:.2f}".format(confidence)
        landmarks = list(map(int, face[4:len(face)-1]))
        landmarks = np.array_split(landmarks, len(landmarks) / 2)

        ### WHENet head pose

        x_min = box[0]
        x_max = x_min + box[2]
        y_min = box[1]
        y_max = y_min + box[3]

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

        # print(f'yaw: {yaw}, pitch: {pitch}, roll: {roll}')

        # Draw
        draw_axis(
            frame,
            yaw,
            pitch,
            roll,
            tdx=(x_min+x_max)/2,
            tdy=(y_min+y_max)/2,
            size=abs(x_max-x_min)//2
        )

        # # # Draw face detected
        # cv2.rectangle(frame, box, (0, 0, 255), 2, cv2.LINE_AA)
        for landmark in landmarks:
            cv2.circle(frame, landmark, 2, (0, 0, 255), -1, cv2.LINE_AA)
        position = (box[0], box[1] - 10)
        cv2.putText(frame, confidence, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    if valid:
        teste += 1
        if teste > 5:
            t1 = time.time()
            delta = (t1-t0)/teste
            t0 = time.time()
            teste = 0

    if delta != 0 and delta > 0.001:
        cv2.putText(frame, f"FPS: {round(1/delta)}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('test', frame)
    # videoWriter.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

