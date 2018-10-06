#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage.filters import threshold_local
from skimage import measure
import numpy as np
import cv2
import imutils

numLastframes = 3

cap = cv2.VideoCapture('/media/sf_ShareFolder/traffic_taiachun.mp4')
#cap = cv2.VideoCapture(0)
# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


def preprocess(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V < T).astype("uint8") * 255
    labels = measure.label(thresh, neighbors=8, background=1)
    mask = np.zeros(thresh.shape, dtype="uint8")

    return thresh

def postprocess(img):
    (T, img) = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    img = cv2.dilate(img, None, iterations=24)
    img = cv2.erode(img, None, iterations=2)
    #img = cv2.dilate(img, None, iterations=12)

    return img

lastFrame = None
ret_last = False
ret_now = False
while(1):
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    next_frame = current_frame + 1
    last_frame = current_frame - numLastframes
    print("{},{},{}".format(next_frame, current_frame, last_frame))

    if last_frame >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
        ret_last, lastFrame = cap.read()

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret_now, frame = cap.read()
    frameOrg = frame.copy()
    if not ret_last:
        lastFrame = frame

    frame = preprocess(frame)
    lastFrame = preprocess(lastFrame)

    fgmask = cv2.absdiff(lastFrame, frame)
    frameFinal = postprocess(fgmask)

    cv2.imshow('fgmask',imutils.resize(frameFinal, width=1024))
    cv2.imshow('frame',imutils.resize(fgmask, width=1024))
    #lastFrame = frame
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()

