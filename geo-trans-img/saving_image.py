import cv2

gray_img = cv2.imread('data/input.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale', gray_img)
cv2.imwrite('data/output.jpg', gray_img)
cv2.waitKey()
