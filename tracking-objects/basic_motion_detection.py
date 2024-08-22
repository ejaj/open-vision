import cv2
import numpy as np

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

BLUR_RADIUS = 21

erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

cap = cv2.VideoCapture(0)
frame = None
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        exit(1)
if frame is not None:
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.GaussianBlur(gray_background,
                                       (BLUR_RADIUS, BLUR_RADIUS), 0)

ret, frame = cap.read()
while ret:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,
                                  (BLUR_RADIUS, BLUR_RADIUS), 0)

    diff = cv2.absdiff(gray_background, gray_frame)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    if OPENCV_MAJOR_VERSION >= 4:
        contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        _, contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

    for c in contours:
        if cv2.contourArea(c) > 4000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('diff', diff)
    cv2.imshow('thresh', thresh)
    cv2.imshow('detection', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
