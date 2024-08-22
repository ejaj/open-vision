import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("data/video/hallway.mpg")
if not cap.isOpened():
    print("Error: Video file couldn't be opened")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Get frame dimensions and set up video writer
frame_height, frame_width = frame.shape[:2]
output_size = (frame_width * 2, frame_height * 2)  # For 2x2 grid
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data/video/output.avi', fourcc, 20.0, output_size)

# Initialize background subtractor and structuring elements
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

while ret:
    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    # Find contours and draw bounding boxes for significant objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0))

    # Prepare images for video output
    background_image = bg_subtractor.getBackgroundImage()
    if background_image is None:
        background_image = np.zeros_like(frame)

    # Convert images to BGR for consistency in output video
    fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Concatenate frames for output
    top_row = cv2.hconcat([fg_mask_bgr, thresh_bgr])
    bottom_row = cv2.hconcat([background_image, frame])
    combined_frame = cv2.vconcat([top_row, bottom_row])

    # Write and display the combined frame
    out.write(combined_frame)
    cv2.imshow('Combined', combined_frame)

    k = cv2.waitKey(50) & 0xFF
    if k == 27:  # ESC key to exit
        break
    ret, frame = cap.read()

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
