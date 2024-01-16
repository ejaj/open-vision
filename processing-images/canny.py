import cv2

img = cv2.imread("data/statue_small.jpg", cv2.IMREAD_GRAYSCALE)
canny_img = cv2.Canny(img, 200, 300)
cv2.imshow("data/canny", canny_img)
cv2.waitKey()
cv2.destroyAllWindows()
