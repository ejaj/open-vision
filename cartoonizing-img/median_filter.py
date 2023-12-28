import cv2
import numpy as np

img = cv2.imread('data/green_dots.png')
output = cv2.medianBlur(img, ksize=7)
cv2.imshow('Input', img)
cv2.imshow('Median filter', output)
cv2.waitKey()
