import numpy as np
import cv2

planets = cv2.imread('data/planet_glow.jpg')
gray = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, 1, 120,
    param1=90, param2=40, minRadius=0, maxRadius=0
)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(planets, (i[0], i[1]), i[2],
                   (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(planets, (i[0], i[1]), 2,
                   (0, 0, 255), 3)

cv2.imshow("HoughCircles", planets)
cv2.waitKey()
cv2.destroyAllWindows()
