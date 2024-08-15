import cv2
import sys

image = cv2.imread('data/my_image.png.png')
if image is None:
    print('Failed to read image from file')
    sys.exit(1)

success = cv2.imwrite('data/my_image.jpg', image)
if not success:
    print('Failed to write image to file')
    sys.exit(1)
