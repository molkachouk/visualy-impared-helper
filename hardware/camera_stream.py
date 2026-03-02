import cv2
import time

class CameraStream:
    def __init__(self, source=0, width=640, height=480, fps_limit=15):
        self.source = source
        self.width = width
        self.height = height
        self.fps_limit = fps_limit
        self.last_time = 0

        self.is_image = isinstance(source, str) and source.lower().endswith(
            ('.png', '.jpg', '.jpeg')
        )

        if self.is_image:
            self.static_frame = cv2.imread(source)
            self.used = False
        else:
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        # FPS limiting
        now = time.time()
        if now - self.last_time < 1 / self.fps_limit:
            return None
        self.last_time = now

        if self.is_image:
            if self.used:
                return None  # avoid repeating same image forever
            self.used = True
            return self.static_frame.copy() if self.static_frame is not None else None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return cv2.resize(frame, (self.width, self.height))

    def stop(self):
        if not self.is_image:
            self.cap.release()
