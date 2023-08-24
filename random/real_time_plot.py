
# import the necessary packages
import cv2
import datetime as dt
import matplotlib.pyplot as plt
import random
import numpy as np

##################################################
### Real time data plotter in opencv (python)  ###
## Plot integer data for debugging and analysis ##
## Contributors - Vinay @ www.connect.vin       ##
## For more details, check www.github.com/2vin  ##
##################################################


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


# Create a plotter class object
p = Plotter(800, 500)

# Create figure for plotting
fig = plt.figure()

# plt.xticks(rotation=45, ha='right')
# plt.subplots_adjust(bottom=0.30)
# plt.title('y vs x')
# plt.ylabel('label')

ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

# def process_data(i, xs, ys):
while True:

    # DADOS
    data = random.randint(0, 100)

    # PLOT
    p.plot(data)

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%M:%S.%f'))
    ys.append(data)

    # Limit x and y lists to 50 items
    xs = xs[-80:]
    ys = ys[-80:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    
