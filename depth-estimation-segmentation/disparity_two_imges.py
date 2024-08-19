import cv2
import numpy as np

minDisparity = 16
numDisparities = 192 - minDisparity
blockSize = 5
uniquenessRatio = 1
speckleWindowSize = 3
speckleRange = 3
disp12MaxDiff = 200
P1 = 600
P2 = 2400

# Create StereoSGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    uniquenessRatio=uniquenessRatio,
    speckleRange=speckleRange,
    speckleWindowSize=speckleWindowSize,
    disp12MaxDiff=disp12MaxDiff,
    P1=P1,
    P2=P2
)

img_left = cv2.imread('data/color1_small.jpg')
img_right = cv2.imread('data/color2_small.jpg')

disparity = stereo.compute(img_left, img_right)

# Normalize disparity for display
disparity_display = cv2.normalize(disparity, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
disparity_display = np.uint8(disparity_display)

# Display images and disparity map
cv2.imshow('Left Image', img_left)
cv2.imshow('Right Image', img_right)
cv2.imshow('Disparity Map', disparity_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
