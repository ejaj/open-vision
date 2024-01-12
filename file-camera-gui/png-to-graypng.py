import cv2
import sys

gray_image = cv2.imread('data/MyPic.png', cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    print('Failed to read image from file')
    sys.exit(1)

success = cv2.imwrite('data/MyPicGray.png', gray_image)
if not success:
    print('Failed to write image to file')
    sys.exit(1)
