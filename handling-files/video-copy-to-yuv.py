import cv2

videoCapture = cv2.VideoCapture(0)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
print(fps)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(size)
videoWriter = cv2.VideoWriter('data/MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
success, frame = videoCapture.read()
while success:
    videoWriter.write(frame)
    success, frame = videoCapture.read()
