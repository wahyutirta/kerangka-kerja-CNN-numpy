# -*- coding: utf-8 -*-
import numpy as np
import cv2 
from matplotlib import pyplot as plt
link = "C:/Users/ASUS/Documents/py/cnn-numpy/data_jepun/Plumeria_rubra_L_tri_color/sudamala_(3).jpg"
img = cv2.imread(link)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#hs
hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

#h
hist_h = cv2.calcHist( [hsv], [0], None, [180], [0, 180] )
hist_h = hist_h.T
cv2.imshow('img', img)


"""
img = cv2.imread(link)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3, scale=1)
y = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=3, scale=1)
absx= cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
cv2.imshow('edge', edge)
cv2.imshow('edgex', x)
cv2.imshow('edgey', y)
cv2.waitKey(0)
cv2.destroyAllWindows()"""