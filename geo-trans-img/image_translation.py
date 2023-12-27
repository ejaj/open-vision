import cv2
import numpy as np

img = cv2.imread('data/input.jpg')
num_rows, num_cols = img.shape[:2]
translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
img_translation = cv2.warpAffine(
    img,
    translation_matrix,
    (num_cols, num_rows),
    cv2.INTER_LINEAR
)

translation_matrix = np.float32([[1, 0, -30], [0, 1, -50]])
img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + 70 + 30, num_rows + 110 + 50))
# img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows), cv2.INTER_LINEAR, cv2.BORDER_WRAP,1)
cv2.imshow('Translation', img_translation)
cv2.waitKey()
