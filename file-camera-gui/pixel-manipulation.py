import cv2 as cv

# Read the image
img = cv.imread('data/1.png')

if img is not None:
    print("Original Pixel Value at (0, 0):", img[0, 0])

    # Change the color of the pixel at (0, 0) to white
    img[0, 0] = [255, 255, 255]

    print("Modified Pixel Value at (0, 0):", img[0, 0])

    # Display the image
    cv.imshow("Pixel manipulation", img)

    # Wait for a key press and then close all windows
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Error: Image not found")
