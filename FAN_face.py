import face_alignment
import cv2
import numpy as np

# load our serialized model from disk
print("[INFO] loading model...")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu', face_detector='blazeface')

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
    ret, img = vs.read()
    frame = cv2.resize(img, (320, 260))

    if not ret:
        continue

    preds = fa.get_landmarks(frame)
    # print(preds)

    if preds is not None:
        preds = np.round(preds).astype(np.int32)[0]
        for ponto in preds:
            x, y = ponto
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('test', frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
