"""
Video frame extraction and face-centered cropping.
"""

import cv2
import numpy as np
from loguru import logger

from .config import CROP_SIZE, IMG_SIZE


def _compute_crop_bounds(center: float, frame_size: int, crop_size: int, axis: str):
    """
    Compute crop bounds that always yield exactly crop_size pixels.
    When face is near border, extend the other side to compensate.

    Example: face_center=1100, frame_size=1296, crop_size=512
      - end = min(1100 + 256, 1296) = 1296 (capped at border)
      - start = 1296 - 512 = 784
      - crop: [784:1296] = 512 pixels
    """
    half = crop_size // 2

    # First, try to extend right/bottom up to frame boundary
    end = min(int(center) + half, frame_size)
    # Then extend left/top to fill crop_size
    start = end - crop_size

    # If start goes negative, cap at 0 and extend right instead
    if start < 0:
        start = 0
        end = crop_size
        logger.debug(f"  {axis}-axis: face near top/left border, "
                     f"extended bottom/right to compensate")

    # If end exceeds frame (frame < crop_size), need padding
    needs_pad = False
    if end > frame_size:
        end = frame_size
        needs_pad = True
        logger.warning(f"  {axis}-axis: frame ({frame_size}) < crop_size ({crop_size}), "
                       f"will pad with black ({crop_size - (end - start)}px)")

    return start, end, needs_pad


def extract_cropped_frames(video_path: str, center_y: float, center_x: float,
                           target_fps: float) -> np.ndarray:
    """
    Extract ALL frames from the entire video at target_fps,
    cropped to CROP_SIZE x CROP_SIZE around the mean face center
    (extending opposite side when face is near border),
    then resized to IMG_SIZE x IMG_SIZE for model input.

    Args:
        video_path: Path to video file
        center_y: Mean face center Y (pixels)
        center_x: Mean face center X (pixels)
        target_fps: Frame rate to extract

    Returns:
        np.ndarray shape (T, IMG_SIZE, IMG_SIZE, 3), dtype uint8
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Frame sampling
    frame_step = max(1.0, video_fps / target_fps)

    # Compute crop bounds - extends opposite side when face is near border
    y0, y1, pad_y = _compute_crop_bounds(center_y, frame_h, CROP_SIZE, "Y")
    x0, x1, pad_x = _compute_crop_bounds(center_x, frame_w, CROP_SIZE, "X")

    face_offset_y = int(center_y) - y0
    face_offset_x = int(center_x) - x0

    logger.info(f"Crop: y=[{y0}:{y1}], x=[{x0}:{x1}] | "
                f"face offset in crop: ({face_offset_y}, {face_offset_x}) | "
                f"video={frame_w}x{frame_h} @ {video_fps:.1f}fps -> extract @ {target_fps}fps | "
                f"crop={CROP_SIZE}x{CROP_SIZE} -> resize={IMG_SIZE}x{IMG_SIZE}")

    frames = []
    frame_idx = 0
    next_sample = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_sample:
            crop = frame[y0:y1, x0:x1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Pad with black if frame is smaller than CROP_SIZE
            if pad_y or pad_x:
                padded = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                h, w = crop_rgb.shape[:2]
                padded[:h, :w] = crop_rgb
                crop_rgb = padded

            # Resize from CROP_SIZE -> IMG_SIZE for model input
            if CROP_SIZE != IMG_SIZE:
                crop_rgb = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            frames.append(crop_rgb)
            next_sample += frame_step

        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    result = np.stack(frames)
    logger.info(f"Extracted {result.shape[0]} frames at {target_fps}fps")
    return result
