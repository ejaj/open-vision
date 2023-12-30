import cv2

# Load the face cascade classifier and the face mask image
face_cascade = cv2.CascadeClassifier('data/cascade_files/haarcascade_frontalface_alt.xml')
face_mask = cv2.imread('data/mask_hannibal.png')

# Check if the cascade classifier is loaded properly
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

# Initialize video capture with the specified camera
cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in face_rects:
        # Adjust the height, width, and y-coordinate of the face rectangle
        h, w = int(1.4 * h), int(1.0 * w)
        y -= int(0.1 * h)

        # Extract the region of interest and resize the mask
        frame_roi = frame[y:y + h, x:x + w]
        face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)

        # Create a mask and its inverse
        gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        try:
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        except cv2.error as e:
            print('Ignoring arithmetic exceptions:', e)
            continue

        # Add the two images to get the final output
        frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)

    cv2.imshow('Face Detector', frame)
    if cv2.waitKey(1) == 27:  # Esc key to break the loop
        break

cap.release()
cv2.destroyAllWindows()
