import numpy as np
import time
import cv2

from retinaface import RetinaFace

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

teste = 0
t0 = 0
t1 = 0
delta = 0
# loop over the frames from the video stream
while True:

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, img = vs.read()
	frame = img
	if not ret:
		continue
	
	valid = False
	detections = RetinaFace.detect_faces(frame)
	print(detections)
	# loop over the detections
	for i in range(0, len(detections)):
		
		identity = detections['face_1']
		score = identity["score"]
		facial_area = identity["facial_area"]
		landmarks = identity["landmarks"]
        
		if score < 0.7:
			continue
		else:
			valid = True
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(score * 100)
		y = facial_area[3] - 10 if facial_area[3] - 10 > 10 else facial_area[3] + 10
		cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
		cv2.putText(frame, text, (facial_area[2], y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		
		
	if valid:
		teste += 1
		if teste > 5:
			t1 = time.time()
			delta = (t1-t0)/teste
			t0 = time.time()
			teste = 0

	if delta != 0 and delta > 0.001:
		cv2.putText(frame, f"FPS: {round(1/delta)}", (0, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# if the `q` key was pressed, break from the loop
	if key == ord("p"):
		cv2.imwrite(f'img-{int(time.time())}.png', img[facial_area[3]: facial_area[1], facial_area[2]: facial_area[0]])

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()




