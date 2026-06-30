"""
Face detection using MediaPipe Tasks API (v0.10+).
Computes mean face center across a video.
"""

import os
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

# Path to short-range face detection model
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "blaze_face_short_range.tflite"
)


def detect_mean_face_center(video_path: str, sample_every_n: int = 5) -> Tuple[float, float]:
    """
    Detect face in sampled frames and return the mean face center (cy, cx).

    Uses MediaPipe short-range face detection (optimized for < 2m, ideal for DMS).

    Args:
        video_path: Path to video file
        sample_every_n: Process every Nth frame (speed vs accuracy tradeoff)

    Returns:
        (mean_center_y, mean_center_x) in pixel coordinates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.debug(f"Face detection: video={frame_w}x{frame_h}, total_frames={total_frames}, sample_every={sample_every_n}")

    centers_x, centers_y = [], []
    frame_idx = 0

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        min_detection_confidence=0.5,
    )

    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                if result.detections:
                    bbox = result.detections[0].bounding_box
                    cx = bbox.origin_x + bbox.width / 2
                    cy = bbox.origin_y + bbox.height / 2
                    centers_x.append(cx)
                    centers_y.append(cy)

            frame_idx += 1

    cap.release()

    if not centers_x:
        logger.warning(f"No face detected in {video_path}, using frame center as fallback")
        return frame_h / 2.0, frame_w / 2.0

    mean_cx = np.mean(centers_x)
    mean_cy = np.mean(centers_y)
    sampled = total_frames // sample_every_n
    logger.info(f"Face detected in {len(centers_x)}/{sampled} sampled frames. "
                f"Mean center: ({mean_cy:.0f}, {mean_cx:.0f})")

    return mean_cy, mean_cx
