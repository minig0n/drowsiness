import os
import numpy as np
import cv2
import time


# Inicialização
directory = os.path.dirname(__file__)
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    exit()

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
