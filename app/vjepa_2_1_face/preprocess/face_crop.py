"""Face-centered cropping preprocessor for the V-JEPA 2.1 8-d training pipeline.

For every source video:
  1. Run MediaPipe face detection on every frame.
  2. Take the MEDIAN face center across frames (robust to per-frame jitter/misses).
  3. Crop a CROP x CROP window centered on that median center, CLAMPED to the frame.
     If the face sits near an edge/corner the window snaps inward, so the face ends
     up off-center in the crop. E.g. center=900, crop=512, frame=1000:
     900-256 = 644, but 644+512 = 1156 > 1000, so x0 = 1000-512 = 488.
  4. Write NUM_FRAMES frames out as an mp4.

Outputs (into --meta-dir):
  faces_<crop>/*.mp4  cropped clips
  centers.json        per-video detection stats + chosen crop box (cache/audit)
  train.csv           "<path> <label>", the space-delimited format that
                      src/datasets/video_dataset.py expects
"""

import argparse
import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np

# MediaPipe's tflite interpreter is not fork-safe, so each worker process builds
# its own detector lazily on first use.
_DETECTOR = None


def _get_detector(model_path: str, min_conf: float):
    global _DETECTOR
    if _DETECTOR is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        opts = vision.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=min_conf,
        )
        _DETECTOR = (vision.FaceDetector.create_from_options(opts), mp)
    return _DETECTOR


def clamp_window(center: float, size: int, extent: int) -> int:
    """Top-left coord of a `size` window centered on `center`, clamped to [0, extent-size]."""
    start = int(round(center - size / 2.0))
    return max(0, min(start, extent - size))


def read_frames(path: str) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def detect_median_center(frames, detector, mp_mod) -> Tuple[float, float, int, int]:
    """Return (cx, cy, n_hits, n_frames). Falls back to frame center if no face is ever found."""
    centers = []
    for fr in frames:
        img = mp_mod.Image(
            image_format=mp_mod.ImageFormat.SRGB,
            data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB),
        )
        res = detector.detect(img)
        if res.detections:
            # largest detection = the subject; ignores smaller background faces
            bb = max(
                res.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            ).bounding_box
            centers.append((bb.origin_x + bb.width / 2.0, bb.origin_y + bb.height / 2.0))

    h, w = frames[0].shape[:2]
    if not centers:
        return float(w) / 2.0, float(h) / 2.0, 0, len(frames)
    cx, cy = np.median(np.asarray(centers, dtype=np.float64), axis=0)
    return float(cx), float(cy), len(centers), len(frames)


def pad_to_min(fr, size: int):
    """Reflect-pad a frame up to `size` if it is smaller than the crop."""
    h, w = fr.shape[:2]
    if h >= size and w >= size:
        return fr
    ph, pw = max(0, size - h), max(0, size - w)
    return cv2.copyMakeBorder(fr, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_REFLECT_101)


def process_one(job) -> Optional[dict]:
    path, out_dir, crop, num_frames, model_path, min_conf, fps = job
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, name + ".mp4")

    try:
        frames = read_frames(path)
        if not frames:
            return {"video": path, "status": "empty"}

        detector, mp_mod = _get_detector(model_path, min_conf)
        cx, cy, hits, total = detect_median_center(frames, detector, mp_mod)

        frames = [pad_to_min(f, crop) for f in frames]
        h, w = frames[0].shape[:2]
        x0 = clamp_window(cx, crop, w)
        y0 = clamp_window(cy, crop, h)

        # temporal: take the centre `num_frames`; loop-pad if the clip is short
        if len(frames) >= num_frames:
            start = (len(frames) - num_frames) // 2
            sel = frames[start : start + num_frames]
        else:
            sel = [frames[i % len(frames)] for i in range(num_frames)]

        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop, crop))
        if not writer.isOpened():
            return {"video": path, "status": "writer_failed"}
        for fr in sel:
            writer.write(fr[y0 : y0 + crop, x0 : x0 + crop])
        writer.release()

        return {
            "video": path,
            "out": out_path,
            "status": "ok",
            "src_size": [w, h],
            "median_center": [cx, cy],
            "crop_xywh": [x0, y0, crop, crop],
            "det_hits": hits,
            "n_frames_src": total,
            "n_frames_out": len(sel),
        }
    except Exception as e:  # noqa: BLE001 - keep the pool alive, report per item
        return {"video": path, "status": f"error: {type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description="Face-centered crop preprocessing")
    ap.add_argument("--videos", default="/home/ubuntu/inwdata/prithvi/videos")
    ap.add_argument("--meta-dir", default="/home/ubuntu/inwdata/prithvi/git/vjepa2/workarea")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument(
        "--model",
        default="/home/ubuntu/inwdata/prithvi/git/vjepa2/workarea/models/blaze_face_short_range.tflite",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = os.path.join(args.meta_dir, f"faces_{args.crop}")
    os.makedirs(out_dir, exist_ok=True)

    vids = sorted(
        os.path.join(args.videos, f)
        for f in os.listdir(args.videos)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    )
    print(f"found {len(vids)} videos -> {out_dir}")

    from p_tqdm import p_umap

    jobs = [(v, out_dir, args.crop, args.num_frames, args.model, args.min_conf, args.fps) for v in vids]
    results = p_umap(process_one, jobs, num_cpus=args.workers, desc="face-crop")

    ok = [r for r in results if r and r.get("status") == "ok"]
    bad = [r for r in results if not r or r.get("status") != "ok"]
    no_face = [r for r in ok if r["det_hits"] == 0]

    centers_path = os.path.join(args.meta_dir, "centers.json")
    with open(centers_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args.meta_dir, "train.csv")
    with open(csv_path, "w") as f:
        for r in sorted(ok, key=lambda r: r["out"]):
            f.write(f"{r['out']} 0\n")

    print(f"ok={len(ok)}  failed={len(bad)}  no_face_fallback={len(no_face)}")
    for r in bad[:10]:
        print("  FAIL", r)
    print(f"wrote {csv_path}\nwrote {centers_path}")


if __name__ == "__main__":
    main()
