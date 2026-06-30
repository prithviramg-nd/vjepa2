"""
Configuration constants for the VJEPA2.1 embedding pipeline.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
VJEPA2_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(VJEPA2_ROOT, "models", "vjepa2_1_vitg_384.pt")
DEFAULT_DOWNLOAD_DIR = os.path.join(VJEPA2_ROOT, "workarea", "temp_downloads")

# ─── Model ───────────────────────────────────────────────────────────────────
IMG_SIZE = 384          # Feed 384x384 directly (no downscale)
CROP_SIZE = 512         # 512 left + 512 right from face center
PATCH_SIZE = 16         # 768/16 = 48 patches per spatial dim
TUBELET_SIZE = 2        # Groups 2 frames temporally
NUM_FRAMES_PER_CHUNK = 64  # VJEPA2.1 max frames per forward pass

# Derived constants
SPATIAL_PATCHES = IMG_SIZE // PATCH_SIZE           # 48
SPATIAL_TOKENS = SPATIAL_PATCHES ** 2              # 2304
TEMPORAL_TOKENS = NUM_FRAMES_PER_CHUNK // TUBELET_SIZE  # 32
TOKENS_PER_CHUNK = TEMPORAL_TOKENS * SPATIAL_TOKENS    # 73,728
EMBED_DIM = 1408  # ViT-g embedding dimension

# ─── Normalization ───────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ─── AVC API ─────────────────────────────────────────────────────────────────
AVC_API_ENDPOINT = "https://analytics-kpis.netradyne.info/avc_api"

# ─── Pipeline defaults ───────────────────────────────────────────────────────
TARGET_FPS_LIST = [20, 10]
FACE_SAMPLE_EVERY_N = 5  # Sample every Nth frame for face detection
CHUNKS_PER_GPU = 4        # Number of chunks to batch per GPU (4 x 4 GPUs = 16 parallel)
