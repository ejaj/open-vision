import cv2 as cv

img = cv.imread('data/1.png')
print(img.item(150, 120, 0))  # Current blue value
img.itemset((150, 120, 0), 255)  # Change blue value
print(img.item(150, 120, 0))  # New blue value
