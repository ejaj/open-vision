import cv2

img = cv2.imread('./data/input.jpg')
cv2.imshow("Input image", img)
cv2.waitKey()
