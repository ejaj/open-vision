import cv2

# Open the video file
videoCapture = cv2.VideoCapture('data/MyInputVid.avi')

if not videoCapture.isOpened():
    print("Error: Unable to open video file.")
else:

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(
        'data/MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    print(fps)
    print(size)
    # Loop to read and display frames<
    while True:
        ret, frame = videoCapture.read()
        if not ret:
            print("Reached end of video")
            break
        cv2.imshow('Video Playback', frame)
        videoWriter.write(frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()