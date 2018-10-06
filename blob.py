# Standard imports
import cv2
import numpy as np;
import imutils
 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

cap = cv2.VideoCapture('/media/sf_ShareFolder/traffic_taiachun.mp4')
#cap = cv2.VideoCapture(0)
# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect blobs.
    keypoints = detector.detect(frame)
 
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(1)

