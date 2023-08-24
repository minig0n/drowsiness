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

# Plot values in opencv program
class Plotter:
    def __init__(self, plot_width, plot_height):
        self.width = plot_width
        self.height = plot_height
        self.color = (255, 0 ,0)
        self.val = []
        self.plot_canvas = np.ones((self.height, self.width, 3))*255

	# Update new values in plot
    def plot(self, val, label = "plot"):
        self.val.append(int(val))
        while len(self.val) > self.width:
            self.val.pop(0)

        self.show_plot(label)

    # Show plot using opencv imshow
    def show_plot(self, label):
        self.plot_canvas = np.ones((self.height, self.width, 3))*255
        cv2.line(self.plot_canvas, (0, int(self.height/2) ), (self.width, int(self.height/2)), (0,255,0), 1)
        for i in range(len(self.val)-1):
            cv2.line(self.plot_canvas, (i, int(self.height/2) - self.val[i]), (i+1, int(self.height/2) - self.val[i+1]), self.color, 1)

        cv2.imshow(label, self.plot_canvas)
        cv2.waitKey(10)


## Test on sample data

# Create a plotter class object
p = Plotter(800, 500)

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

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

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 5
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# def process_data(i, xs, ys):
while True:

    ret, frame = cap.read()

    if frame is not None:
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = detector(gray)
        for face in faces:
            # extract the coordinates of the bounding box
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # apply the shape predictor to the face ROI
            if face is not None:
                
                shape = predictor(gray, face)
                
                for n in range(0, 68):
                    x = shape.part(n).x
                    y = shape.part(n).y
                    # cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

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
                
                for landmark in right_eye:
                    cv2.circle(frame, landmark, 4, (255, 0, 0), -1)

                for landmark in left_eye:
                    cv2.circle(frame, landmark, 4, (255, 0, 0), -1)

                EAR = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye))

                data = EAR

                p.plot(data*1000 - 500)

                cv2.imshow("FRAME", frame)
                if cv2.waitKey(1)&0xFF==ord("q"): 
                    break

                # # Add x and y to lists
                xs.append(dt.datetime.now().strftime('%M:%S.%f'))
                ys.append(data)

                # Limit x and y lists to 50 items
                xs = xs[-80:]
                ys = ys[-80:]

                # Draw x and y lists
                ax.clear()
                ax.plot(xs, ys)

                # Format plot
                plt.xticks(rotation=45, ha='right')
                plt.subplots_adjust(bottom=0.30)
                plt.title('EAR vs time')
                plt.ylabel('EAR')

            #     # print(eye)
            
            


# Set up plot to call animate() function periodically

import time
time.sleep(2)
ani = animation.FuncAnimation(fig, process_data, fargs=(xs, ys), interval=20)
plt.show(block=True)