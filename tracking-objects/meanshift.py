import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

# Capture several frames to allow the camera's autoexposure to adjust.
for i in range(10):
    ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit(1)

# Define an initial tracking window in the center of the frame.
frame_h, frame_w = frame.shape[:2]
w = frame_w // 8
h = frame_h // 8
x = frame_w // 2 - w // 2
y = frame_h // 2 - h // 2

track_window = (x, y, w, h)

# Calculate the normalized HSV histogram of the initial window.
roi = frame[y:y + h, x:x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = None
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Plot the color histogram of the ROI (Hue Channel)
hist = roi_hist.flatten()

# Create an array with the corresponding colors for each hue value
hue_values = np.arange(180)
color_bar = np.zeros((50, 180, 3), dtype="uint8")
for i, hue in enumerate(hue_values):
    color_bar[:, i] = [hue, 255, 255]

color_bar = cv2.cvtColor(color_bar, cv2.COLOR_HSV2BGR)
plt.figure(figsize=(10, 4))
plt.bar(hue_values, hist, color=color_bar[0] / 255.0, edgecolor='none')
plt.xlim([0, 180])
plt.title('Color Histogram (Hue Channel)')
plt.xlabel('Hue Value')
plt.ylabel('Frequency')
plt.show()

# Define the termination criteria:
# 10 iterations or convergence within 1-pixel radius.
term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

ret, frame = cap.read()
while ret:
    # Perform back-projection of the HSV histogram onto the frame.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Perform tracking with MeanShift.
    num_iters, track_window = cv2.meanShift(back_proj, track_window, term_crit)

    # Draw the tracking window.
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('back-projection', back_proj)
    cv2.imshow('meanshift', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    ret, frame = cap.read()
