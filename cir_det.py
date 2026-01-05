import numpy as np
import cv2 as cv

img_location = 'circles.png'
#GET IMAGE FROM SOURCE< AND PRE PROCESS
img = cv.imread(img_location)
if img is None:
	print('no img found')
og= cv.imread(img_location)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img = cv.blur(img,(5,5))
#FIND CIRCLES
circles= cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,minRadius=0,maxRadius=0)
#Copied from https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
	cv.circle(og,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
	cv.circle(og,(i[0],i[1]),2,(0,0,255),3)
 
cv.imshow('detected circles',og)
cv.waitKey(0)
