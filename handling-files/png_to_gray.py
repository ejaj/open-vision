import cv2
import sys

grayImage = cv2.imread('data/my_image.png', cv2.IMREAD_GRAYSCALE)
if grayImage is None:
    print('Failed to read image from file')
    sys.exit(1)

success = cv2.imwrite('data/MyPicGray.png', grayImage)
if not success:
    print('Failed to write image to file')
    sys.exit(1)
