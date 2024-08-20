import cv2
from matplotlib import pyplot as plt

img_1 = cv2.imread('data/nasa_logo.png', cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread('data/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE)
# Perform ORB feature detection and description.
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img_1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img_2, None)

# Perform brute-force KNN matching.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Sort the pairs of matches by distance.
pairs_of_matches = sorted(pairs_of_matches, key=lambda x: x[0].distance)
# Draw the 25 best pairs of matches.
img_pairs_of_matches = cv2.drawMatchesKnn(
    img_1, keypoints1, img_2, keypoints2, pairs_of_matches[:25], img_2,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_pairs_of_matches)
plt.show()

# Apply the ratio test.
matches = [x[0] for x in pairs_of_matches
           if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]

# Draw the best 25 matches.
img_matches = cv2.drawMatches(
    img_1, keypoints1, img_2, keypoints2, matches[:25], img_2,
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Show the matches.
plt.imshow(img_matches)
plt.show()
