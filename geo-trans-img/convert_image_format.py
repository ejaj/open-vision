import cv2

img = cv2.imread('data/input.jpg')
cv2.imwrite('data/output.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
