import cv2


def print_howto():
    print("""
        Change color space of the
        input video stream using keyboard controls. The control keys are: 
            1. Grayscale - press 'g'
            2. YUV - press 'y'
            3. HSV - press 'h'
    """)


# print_howto()


if __name__ == '__main__':
    print_howto()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam. Make sure the device is connected and accessible.")
    try:
        cur_mode = None
        while True:
            # Read the current frame from webcam
            ret, frame = cap.read()
            # Resize the captured image
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            c = cv2.waitKey(1)
            if c == 27:
                break

            if c != -1 and c != 255 and c != cur_mode:
                cur_mode = c
            print(cur_mode)
            if cur_mode == ord('g'):
                output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif cur_mode == ord('y'):
                output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            elif cur_mode == ord('h'):
                output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            else:
                output = frame
            cv2.imshow('Webcam', output)

    except Exception as e:
        print(e)
    finally:
        cap.release()
        cv2.destroyAllWindows()
