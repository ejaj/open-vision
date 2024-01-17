import cv2
import numpy as np

# StereoSGBM Parameters
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

# Load stereo images
imgL = cv2.imread('data/color1_small.jpg')
imgR = cv2.imread('data/color2_small.jpg')

# Compute disparity
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Normalize for display
disparity_normalized = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                     dtype=cv2.CV_8U)
# Display images and disparity map
cv2.imshow('Left Image', imgL)
cv2.imshow('Right Image', imgR)
cv2.imshow('Disparity Map', disparity_normalized)

# Wait for a key press
cv2.waitKey()
cv2.destroyAllWindows()
