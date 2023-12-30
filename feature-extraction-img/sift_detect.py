import cv2

input_image = cv2.imread('data/fishing_house.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
assert hasattr(cv2, 'xfeatures2d'), "xfeatures2d is not available"
sift = cv2.SIFT_create()
keypoints = sift.detect(gray_image, None)

cv2.drawKeypoints(input_image, keypoints, input_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT features', input_image)
cv2.waitKey()
