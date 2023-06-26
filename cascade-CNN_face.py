# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

import cv2
 
# # draw an image with detected objects
# def draw_image_with_boxes(filename, result_list):
#  # load the image
#  data = pyplot.imread(filename)
#  # plot the image
#  pyplot.imshow(data)
#  # get the context for drawing boxes
#  ax = pyplot.gca()
#  # plot each box
#  for result in result_list:
#  # get coordinates
#  x, y, width, height = result['box']
#  # create the shape
#  rect = Rectangle((x, y), width, height, fill=False, color='red')
#  # draw the box
#  ax.add_patch(rect)
#  # draw the dots
#  for key, value in result['keypoints'].items():
#  # create and draw dot
#  dot = Circle(value, radius=2, color='red')
#  ax.add_patch(dot)
#  # show the plot
#  pyplot.show()
 
# filename = 'test1.jpg'
# # load image from file
# pixels = pyplot.imread(filename)
# # create the detector, using default weights
# detector = MTCNN()
# # detect faces in the image
# faces = detector.detect_faces(pixels)
# # display faces on the original image
# draw_image_with_boxes(filename, faces)


cap = cv2.VideoCapture(0)

detector = MTCNN()

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (300, 300))
    img = frame
    if ret:
        faces = detector.detect_faces(frame)
        
        for face in faces:
            # print(face)
            box = face['box']
            landmark = face['keypoints']
            # cv2.rectangle(frame, (box[1], box[2]), (box[0], box[3]), (0, 0, 255), 2)
            cv2.circle(frame, landmark['left_eye'], 4, (255, 0, 0), -1)
            cv2.circle(frame, landmark['right_eye'], 4, (255, 0, 0), -1)
            cv2.circle(frame, landmark['nose'], 4, (255, 0, 0), -1)
            cv2.circle(frame, landmark['mouth_left'], 4, (255, 0, 0), -1)
            cv2.circle(frame, landmark['mouth_right'], 4, (255, 0, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cap.release()