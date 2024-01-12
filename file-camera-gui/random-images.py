import cv2
import numpy as np
import os

# Make an array of 120,000 random bytes.
random_byte_array = bytearray(os.urandom(120000))
flat_number_array = np.array(random_byte_array)

# Convert the array to make a 400x300 grayscale image.

gray_image = flat_number_array.reshape(300, 400)
cv2.imwrite('data/random_gray.png', gray_image)

# Convert the array to make a 400x100 color image.
bgr_image = flat_number_array.reshape(100, 400, 3)
cv2.imwrite('data/random_color.png', bgr_image)
