import cv2
import depth
from managers import WindowManager, CaptureManager
import filters


class Cameo(object):

    def __init__(self):
        self._window_manager = WindowManager('Cameo', self.onKeypress)
        self._capture_manager = CaptureManager(cv2.VideoCapture(0), self._window_manager, True)
        self._curve_filter = filters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop."""
        self._window_manager.create_window()
        while self._window_manager.is_window_created:
            self._capture_manager.enter_frame()
            frame = self._capture_manager.frame

            if frame is not None:
                filters.stroke_edges(frame, frame)
                self._curve_filter.apply(frame, frame)

            self._capture_manager.exit_frame()
            self._window_manager.process_events()

    def onKeypress(self, keycode):
        """
        Handle a keypress.
        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        escape -> Quit.

        """
        if keycode == 32:  # space
            self._capture_manager.write_image('data/screenshot.png')
        elif keycode == 9:  # tab
            if not self._capture_manager.is_writing_video:
                self._capture_manager.start_writing_video('data/screencast.avi')
            else:
                self._capture_manager.stop_writing_video()
        elif keycode == 27:  # escape
            self._window_manager.destroy_window()


class CameoDepth(Cameo):

    def __init__(self):
        self._window_manager = WindowManager('Cameo', self.onKeypress)
        # device = cv2.CAP_OPENNI2 # uncomment for Microsoft Kinect via OpenNI2
        device = cv2.CAP_OPENNI2_ASUS  # uncomment for Asus Xtion or Occipital Structure via OpenNI2
        self._capture_manager = CaptureManager(
            cv2.VideoCapture(device), self._window_manager, True)
        self._curve_filter = filters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop."""
        self._window_manager.create_window()
        while self._window_manager.is_window_created:
            self._capture_manager.enter_frame()
            self._capture_manager.channel = cv2.CAP_OPENNI_DISPARITY_MAP
            disparity_map = self._capture_manager.frame
            self._capture_manager.channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
            valid_depth_mask = self._capture_manager.frame
            self._capture_manager.channel = cv2.CAP_OPENNI_BGR_IMAGE
            frame = self._capture_manager.frame
            if frame is None:
                # Failed to capture a BGR frame.
                # Try to capture an infrared frame instead.
                self._capture_manager.channel = cv2.CAP_OPENNI_IR_IMAGE
                frame = self._capture_manager.frame

            if frame is not None:

                # Make everything except the median layer black.
                mask = depth.create_median_mask(disparity_map, valid_depth_mask)
                frame[mask == 0] = 0

                if self._capture_manager.channel == cv2.CAP_OPENNI_BGR_IMAGE:
                    # A BGR frame was captured.
                    # Apply filters to it.
                    filters.stroke_edges(frame, frame)
                    self._curve_filter.apply(frame, frame)

            self._capture_manager.exit_frame()
            self._window_manager.process_events()


if __name__ == "__main__":
    CameoDepth().run()
