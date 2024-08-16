import cv2
import numpy as np
import time


class CaptureManager:
    def __init__(self, capture, live_preview=None, mirror_preview=False):
        self.live_preview = live_preview
        self.mirror_preview = mirror_preview
        self._capture = capture
        self._channel = 0
        self._entered_frame = False
        self._frame = None
        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = 0
        self._fps_estimate = None
        self._grayscale = False

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._entered_frame and self._frame is None:
            _, self._frame = self._capture.retrieve(self._frame, self.channel)
        return self._frame

    @property
    def is_writing_image(self):
        return self._image_filename is not None

    @property
    def is_writing_video(self):
        return self._video_filename is not None

    def enter_frame(self):
        assert not self._entered_frame, 'previous enter_frame() had no matching exit_frame()'

        if self._capture is not None:
            self._entered_frame = self._capture.grab()

    def exit_frame(self):
        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self._frame is None:
            self._entered_frame = False
            return
        # Update the FPS estimate and related variables.
        if self._frames_elapsed == 0:
            self._start_time = time.perf_counter()
        else:
            time_elapsed = time.perf_counter() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1

        # Apply grayscale if the flag is set
        if self._grayscale:
            self._frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)

        # Draw to the window, if any.
        if self.live_preview is not None:
            if self.mirror_preview:
                mirrored_frame = np.fliplr(self._frame)
                self.live_preview.show(mirrored_frame)
            else:
                self.live_preview.show(self._frame)

        # Write to the image file, if any.
        if self.is_writing_image:
            cv2.imwrite(self._image_filename, self._frame)
            self._image_filename = None
        # Write to the video file, if any.
        self._write_video_frame()

        # Release the frame.
        self._frame = None
        self._entered_frame = False

    def write_image(self, filename):
        self._image_filename = filename

    def start_writing_video(self, filename, encoding=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')):
        self._video_filename = filename
        self._video_encoding = encoding

    def stop_writing_video(self):
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return
        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if np.isnan(fps) or fps <= 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._frames_elapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._video_writer = cv2.VideoWriter(self._video_filename, self._video_encoding, fps, size)
        self._video_writer.write(self._frame)

    def make_grayscale(self, grayscale):
        self._grayscale = grayscale


class WindowManager:
    def __init__(self, window_name, keypressCallback=None):
        self._window_name = window_name
        self._keypressCallback = keypressCallback
        self._window_created = False  #

    @property
    def is_window_created(self):
        return self._window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._window_created = True

    def show(self, frame):
        cv2.imshow(self._window_name, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._window_name)
        self._window_created = False

    def process_events(self):
        keycode = cv2.waitKey(1)
        if self._keypressCallback is not None and keycode != -1:
            self._keypressCallback(keycode)
