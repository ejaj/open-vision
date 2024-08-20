import cv2

img = cv2.imread('data/varese.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(8000)

keypoints, descriptors = surf.detectAndCompute(gray, None)


img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (51, 163, 236),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SURF Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
