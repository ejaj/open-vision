import cv2
import numpy as np

# Create a blank 200x200 black image
img = np.zeros((200, 200), dtype=np.uint8)

# Insert a white square in the middle
img[50:150, 50:150] = 255
img[70:150, 70:150] = 128
cv2.imshow("Original", img)

# Apply threshold
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Copy the original image and convert to color for contour drawing
color = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

# Draw contours
img_with_contours = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow("Contours", img_with_contours)
cv2.waitKey()
cv2.destroyAllWindows()
