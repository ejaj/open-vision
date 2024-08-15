import cv2

clicked = False


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True


cap = cv2.VideoCapture(0)
cv2.namedWindow('My Window')
cv2.setMouseCallback('My Window', on_mouse)

print('Showing camera feed. Click window or press any key to stop.')
success, frame = cap.read()
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('My Window', frame)
    success, frame = cap.read()
cap.release()
cv2.destroyAllWindows()
