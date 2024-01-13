import cv2

clicked = False


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN and flags == cv2.EVENT_FLAG_LBUTTON:
        clicked = True


capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Camera is not accessible")
    exit()

cv2.namedWindow('My Window')
cv2.setMouseCallback('My Window', on_mouse)

print('Showing camera feed. Click window or press any key to stop.')

while True:
    success, frame = capture.read()
    if not success:
        break

    cv2.imshow("My Window", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')] or clicked:  # Exit on ESC or 'q' key or mouse click
        break

capture.release()
cv2.destroyAllWindows()
