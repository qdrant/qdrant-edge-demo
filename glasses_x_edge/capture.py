import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .constants import JPEG_QUALITY

FRAME_RESIZE_TARGET = (128, 128)
SSIM_DATA_RANGE = 255

logger = logging.getLogger(__name__)


class VideoCapture:
    def __init__(self, source: str, fps: float):
        self.source = source
        self.fps = fps
        self.capture = None
        self.frame_interval = 1.0 / fps if fps > 0 else 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            logger.error(f"Failed to open video: {self.source}")
            raise RuntimeError(f"Failed to open video: {self.source}")

    def stop(self):
        if self.capture:
            self.capture.release()
            self.capture = None

    def capture_continuous(self):
        video_fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        frames_to_skip = max(1, int(video_fps * self.frame_interval))

        frame_count = 0
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frames_to_skip == 1:
                yield frame
                time.sleep(self.frame_interval)

    @staticmethod
    def calculate_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
        if frame1 is None or frame2 is None:
            return 0.0

        f1 = cv2.resize(frame1, FRAME_RESIZE_TARGET)
        f2 = cv2.resize(frame2, FRAME_RESIZE_TARGET)

        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        score = structural_similarity(g1, g2, data_range=SSIM_DATA_RANGE)
        return float(score)

    def save_frame(self, frame: np.ndarray, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image.save(output_path, "JPEG", quality=JPEG_QUALITY)
        return output_path
