# Importing libraries
import cv2
import numpy as np
# Capturing the video file 0 for videocam else you can provide the url
# capture = cv2.VideoCapture("video_file.avi")

cap = cv2.VideoCapture(0)
 
# Till you scan the video
while(1):

    # Reading the first frame
    suc1, frame1 = cap.read()

    # Capture another frame and convert to gray scale
    suc2, frame2 = cap.read()

    if suc1 and suc2:

        # Create mask
        hsv_mask = np.zeros_like(frame1)
        # Make image saturation to a maximum value
        hsv_mask[..., 1] = 255

        # Convert to gray scale
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
        # Optical flow is now calculated
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnite and angle of 2D vector
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to rgb
        rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    
        cv2.imshow('frame2', rgb_representation)
        if cv2.waitKey(1)&0xFF == ord("q"): 
            break
        
        prvs = next
            
 
cap.release()
cv2.destroyAllWindows()