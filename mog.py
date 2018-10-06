#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import imutils

minArea = 1000
lineBold = 2

cap = cv2.VideoCapture('/media/sf_ShareFolder/traffic_taiachun.mp4')
#cap = cv2.VideoCapture(0)
# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float


#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorMOG2(history=600, detectShadows=True)

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
    boundcolor = (0,0,255)
    cv2.putText(img, str(lanA), (34, 556), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (0,558), (228, 536), (250, 0, 0), lineBold)
    cv2.putText(img, str(lanB), (264, 474), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (140,528), (568,434), (0, 250, 0), lineBold)

    cv2.putText(img, str(lanC), (1210, 404), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (1368,406), (1106, 364), (251, 0, 0), lineBold)
    cv2.putText(img, str(lanD), (1560, 446), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (1392,418), (1700,492), (0, 251, 0), lineBold)

    cv2.putText(img, str(lanE), (1464, 698), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (1516,628), (1366,770), (252, 0, 0), lineBold)
    cv2.putText(img, str(lanF), (1316, 836), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (1378,772), (1224,916), (0, 252, 0), lineBold)

    cv2.putText(img, str(lanG), (812, 884), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (448,850), (1074,984), (253, 0, 0), lineBold)
    cv2.putText(img, str(lanH), (224, 780), cv2.FONT_HERSHEY_COMPLEX, 1.6, boundcolor, 2)
    cv2.line(img, (0,710), (454,828), (0, 253, 0), lineBold)

    return img

def caculateCars(img, x, y):
    global lanA, lanB, lanC, lanD, lanE, lanF, lanG, lanH

    color = img[y,x]
    dir = ""

    if(color==[250, 0, 0]).all():
        dir = "left-leave"
        lanA += 1
    elif(color==[0,250,0]).all():
        dir = "left-enter"
        lanB += 1
    elif(color==[251,0,0]).all():
        dir = "top-leave"
        lanC += 1
    elif(color==[0,251,0]).all():
        dir = "top-enter"
        lanD += 1
    elif(color==[252,0,0]).all():
        dir = "right-enter"
        lanE += 1
    elif(color==[0,252,0]).all():
        dir = "right-leave"
        lanF += 1
    elif(color==[253,0,0]).all():
        dir = "bottom-leave"
        lanG += 1
    elif(color==[0,253,0]).all():
        dir = "bottom_enter"
        lanH += 1

    print(dir)
    return dir

lanA = 0
lanB = 0
lanC = 0
lanD = 0
lanE = 0
lanF = 0
lanG = 0
lanH = 0
while(1):
    ret, frame = cap.read()
    frameOrg = frame.copy()
    frame = preprocess(frame)
    
    fgmask = fgbg.apply(frame)
    fgmask = postprocess(fgmask)

    cnts = findContours(fgmask)
    QttyOfContours = 0

    frameOrg = drawLine(frameOrg)
   #check all found countours
    for c in cnts:
        #if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < minArea:
            continue

        QttyOfContours = QttyOfContours+1    

        #draw an rectangle "around" the object
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frameOrg, (x, y), (x + w, y + h), (0,255,0), 1)
        #cv2.rectangle(fgmask, (x, y), (x + w, y + h), (0, 0, 255), 1)

        #find object's centroid
        CoordXCentroid = int((x+x+w)/2)
        CoordYCentroid = int((y+y+h)/2)
        ObjectCentroid = (CoordXCentroid,CoordYCentroid)
        #cv2.circle(frameOrg, ObjectCentroid, 1, (0, 0, 0), 5)
        #cv2.circle(fgmask, ObjectCentroid, 1, (0, 0, 0), 5)

        direction = caculateCars(frameOrg, CoordXCentroid, CoordYCentroid)
        if(len(direction)>0):
            cv2.rectangle(frameOrg, (x, y), (x + w, y + h), (0, 0, 255), 3)

    #overlay = overlayArea(width, height)
    #frameOrg = cv2.addWeighted(frameOrg,0.7,overlay,0.3,0) 

    cv2.imshow('Original',imutils.resize(frameOrg, width=1024))
    #cv2.imshow('Processed',imutils.resize(fgmask, width=1024))
    

 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()

