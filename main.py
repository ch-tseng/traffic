#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from yoloOpencv import opencvYOLO
from objectTracking import trackingObj
import numpy as np
import cv2
import imutils

minArea = 1000
lineBold = 2
tracking = "CSRT"
yolo = opencvYOLO(modeltype="yolov3", objnames="../../darknet/data/coco.names", 
    weights="../../darknet/weights/yolov3.weights",
    cfg="../../darknet/cfg/yolov3.cfg")


cap = cv2.VideoCapture('/media/sf_ShareFolder/traffic_taiachun.mp4')
#cap = cv2.VideoCapture(0)
FILE_OUTPUT = '/media/sf_ShareFolder/taichun_traffic.avi'

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 30.0, (int(width),int(height)))


def transparentOverlay(src , overlay , pos=(0,0),scale = 1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src


def overlayArea(w, h):
    # loop over the alpha transparency values
    overlay = np.zeros((int(h), int(w), 3), dtype = "uint8")

    pts = np.array([[890,440],[1540,557], [1060,1017],[0,650]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(overlay,[pts],color=(0,255,0))

    return overlay

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (19, 19), 0)

    return img

def postprocess(img):
    (T, img) = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    img = cv2.dilate(img, None, iterations=12)
    img = cv2.erode(img, None, iterations=8)
    img = cv2.dilate(img, None, iterations=4)

    return img

def findContours(img):
    _, cnts, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts

def drawLine(img):
    cv2.line(img, (0,578), (144, 530), (250, 0, 0), lineBold)
    cv2.line(img, (106,516), (480,430), (0, 250, 0), lineBold)

    cv2.line(img, (1016,416), (1268,468), (251, 0, 0), lineBold)
    cv2.line(img, (1300,502), (1524,558), (0, 251, 0), lineBold)

    cv2.line(img, (1508,640), (1388,748), (252, 0, 0), lineBold)
    cv2.line(img, (1376,766), (1246,894), (0, 252, 0), lineBold)

    cv2.line(img, (1130,928), (632,816), (253, 0, 0), lineBold)
    cv2.line(img, (592,804), (90,714), (0, 253, 0), lineBold)

    return img

def caculateCars(img, x, y):
    color = img[y,x]
    dir = ""

    if(color==[250, 0, 0]).all():
        dir = "left-leave"
    elif(color==[0,250,0]).all():
        dir = "left-enter"
    elif(color==[251,0,0]).all():
        dir = "top-leave"
    elif(color==[0,251,0]).all():
        dir = "top-enter"
    elif(color==[252,0,0]).all():
        dir = "right-enter"
    elif(color==[0,252,0]).all():
        dir = "right-leave"
    elif(color==[253,0,0]).all():
        dir = "bottom-leave"
    elif(color==[0,253,0]).all():
        dir = "bottom_enter"

    print(dir)
    return dir

while(1):
    ret, frame = cap.read()

    frameDisplay = yolo.getObject(frame, labelWant="", drawBox=False)
    frameDisplay = drawLine(frameDisplay)

    cv2.imshow('Original',imutils.resize(frameDisplay, width=1024))

    countTotal = 0
    if(yolo.objCounts>0):
        bboxes = yolo.bbox
        tracker = trackingObj()

        if(len(bboxes)>0):
            tracker.createTrackers(frame, bboxes, bold=3, color=(0,0,255), trackerType=tracking)
            success = True
        else:
            success = False

        while success:
            ret, frame = cap.read()
            (success, boxes, frame2) = tracker.updateTrackers(frame, drawbox=False)
            frame2 = drawLine(frame2)

            for id, newbox in enumerate(boxes):
                centerX = int(newbox[0] + (newbox[2]/2))
                centerY = int(newbox[1] + (newbox[3]/2))
                direction = caculateCars(frame2, centerX, centerY)
                if(len(direction)>0):
                    if(tracker.counted[id] == False):
                        countTotal += 1
                        tracker.counted[id] = True
                        tracker.bold = 1
                        print("Counted!")


            cv2.imshow('Original',imutils.resize(frame2, width=1024))


            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
    

cap.release()
cv2.destroyAllWindows()

