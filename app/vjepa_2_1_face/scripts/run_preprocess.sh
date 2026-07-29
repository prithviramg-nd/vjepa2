#!/usr/bin/env bash
# Step 1: face-centered cropping of the raw videos.
# Detects faces with MediaPipe, takes the median face center per video, and crops
# a 512x512 window around it (clamped to the frame edges).
set -euo pipefail

REPO="/home/ubuntu/inwdata/prithvi/git/vjepa2"
PY="${REPO}/venv/bin/python3"
MODEL="${REPO}/workarea/models/blaze_face_short_range.tflite"

if [ ! -f "${MODEL}" ]; then
  echo "downloading MediaPipe face detector..."
  mkdir -p "${REPO}/workarea/models"
  curl -sSL -o "${MODEL}" \
    https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
fi

"${PY}" "${REPO}/app/vjepa_2_1_face/preprocess/face_crop.py" \
  --videos "${1:-/home/ubuntu/inwdata/prithvi/videos}" \
  --meta-dir "${REPO}/workarea" \
  --crop 512 \
  --num-frames 30 \
  --fps 10 \
  --workers 8
