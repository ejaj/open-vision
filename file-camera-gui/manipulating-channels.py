import cv2 as cv

img = cv.imread('data/1.png')
img[:, :, 1] = 0  # Zero out the green channel

# Display the image
cv.imshow("Manipulating Channels", img)
# Wait for a key press and then close all windows
cv.waitKey(0)
cv.destroyAllWindows()
