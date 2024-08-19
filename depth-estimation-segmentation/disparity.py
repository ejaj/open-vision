import numpy as np
import cv2

minDisparity = 16
numDisparities = 192 - minDisparity
blockSize = 5
uniquenessRatio = 1
speckleWindowSize = 3
speckleRange = 3
disp12MaxDiff = 200
P1 = 600
P2 = 2400

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


def update(sliderValue=0):
    try:
        blockSize = cv2.getTrackbarPos('blockSize', 'Disparity')
        uniquenessRatio = cv2.getTrackbarPos(
            'uniquenessRatio', 'Disparity')
        speckleWindowSize = cv2.getTrackbarPos(
            'speckleWindowSize', 'Disparity')
        speckleRange = cv2.getTrackbarPos(
            'speckleRange', 'Disparity')
        disp12MaxDiff = cv2.getTrackbarPos(
            'disp12MaxDiff', 'Disparity')
    except cv2.error:
        # One or more of the sliders has not been created yet.
        return

    stereo.setBlockSize(blockSize)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setSpeckleRange(speckleRange)
    stereo.setDisp12MaxDiff(disp12MaxDiff)

    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    cv2.imshow('Left', img_left)
    cv2.imshow('Right', img_right)
    cv2.imshow('Disparity',
               (disparity - minDisparity) / numDisparities)


cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
cv2.createTrackbar('blockSize', 'Disparity', blockSize, 21,
                   update)
cv2.createTrackbar('uniquenessRatio', 'Disparity',
                   uniquenessRatio, 50, update)
cv2.createTrackbar('speckleWindowSize', 'Disparity',
                   speckleWindowSize, 200, update)
cv2.createTrackbar('speckleRange', 'Disparity',
                   speckleRange, 50, update)
cv2.createTrackbar('disp12MaxDiff', 'Disparity',
                   disp12MaxDiff, 250, update)
update()
cv2.waitKey(0)
cv2.destroyAllWindows()
