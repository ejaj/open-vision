import cv2
from matplotlib import pyplot as plt

img_1 = cv2.imread('data/nasa_logo.png', cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread('data/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE)

# Perform ORB feature detection and description.
orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img_1, None)
kp1, des1 = orb.detectAndCompute(img_2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des0, des1)

matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(
    img_1, kp0, img_2, kp1, matches[:25], img_2,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.show()
