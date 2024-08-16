import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# determine upper and lower HSV limits for (my) skin tones
lower = np.array([0, 100, 0], dtype="uint8")
upper = np.array([50, 255, 255], dtype="uint8")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # switch to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # find mask of pixels within HSV range
    skin_mask = cv2.inRange(hsv, lower, upper)
    # denoise
    skinMask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
    # kernel for morphology operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    # CLOSE (dilate / erode)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # denoise the mask
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
    # only display the masked pixels
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    cv2.imshow("HSV", skin)
    # Check for the 'q' key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
