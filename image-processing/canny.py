import cv2
import numpy as np

img = cv2.imread('data/statue_small.jpg')
canny = cv2.Canny(img, 200, 300)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
