import cv2 as cv

img = cv.imread('data/2.png')
my_roi = img[0:200, 0:200]
img[300:500, 300:500] = my_roi  # Copy ROI to a new location
# Display the image
cv.imshow("Regions of Interest (ROI)", img)
# Wait for a key press and then close all windows
cv.waitKey(0)
cv.destroyAllWindows()
