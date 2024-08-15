import cv2

cap = cv2.VideoCapture(0)
fps = 30
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

out = cv2.VideoWriter("data/my_10_sec.avi", fourcc, fps, size)
success, frame = cap.read()
number_frames = 10 * fps - 1  # 10 sec of frames
while number_frames > 0:
    if frame is not None:
        out.write(frame)
    success, frame = cap.read()
    number_frames = number_frames - 1

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
