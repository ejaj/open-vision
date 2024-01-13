import cv2

cameraCapture = cv2.VideoCapture(0)
if not cameraCapture.isOpened():
    print("Error: Camera not accessible")
    exit()

fps = 30
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
    'data/MyOutputVid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

numFramesRemaining = 10 * fps  # 10 seconds of frames
while numFramesRemaining > 0:
    success, frame = cameraCapture.read()
    if success:
        videoWriter.write(frame)
        numFramesRemaining -= 1
    else:
        break

cameraCapture.release()
videoWriter.release()
