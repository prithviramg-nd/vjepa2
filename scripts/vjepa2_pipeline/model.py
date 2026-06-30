"""
VJEPA2.1 model loading and data-parallel chunked inference.

Loads the full model on each GPU and processes multiple chunks in parallel
(one chunk per GPU). With 4 GPUs, processes 4 chunks simultaneously.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .config import (
    CHUNKS_PER_GPU,
    EMBED_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_FRAMES_PER_CHUNK,
    TOKENS_PER_CHUNK,
    VJEPA2_ROOT,
)

sys.path.insert(0, VJEPA2_ROOT)
from src.models.vision_transformer import vit_giant_xformers_rope  # noqa: E402


# ─── Model Wrapper ───────────────────────────────────────────────────────────

class SingleGPUViT(nn.Module):
    """Runs ViT-g on a single GPU in float32."""

    def __init__(self, model, gpu_id=0):
        super().__init__()
        self.model = model.to(f'cuda:{gpu_id}')
        self.device = f'cuda:{gpu_id}'
        self.gpu_id = gpu_id

    @torch.inference_mode()
    def forward(self, x):
        m = self.model
        x = x.to(device=self.device, dtype=torch.float32)

        if x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // m.tubelet_size
        else:
            _, _, H, W = x.shape
            T = 1
        H_p = H // m.patch_size
        W_p = W // m.patch_size
        if not m.handle_nonsquare_inputs:
            T = H_p = W_p = None

        x = m.patch_embed(x) if m.use_rope else (m.patch_embed(x) + m.interpolate_pos_encoding(x, m.pos_embed))

        for blk in m.blocks:
            x = blk(x, mask=None, attn_mask=None, T=T, H_patches=H_p, W_patches=W_p)

        if m.norm is not None:
            x = m.norm(x)

        return x


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_model(model_path: str, gpu_ids: List[int]):
    """
    Load VJEPA2.1 ViT-g on each GPU for data-parallel inference.
    Returns a list of model wrappers (one per GPU).
    """
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Config: {IMG_SIZE}x{IMG_SIZE}, {NUM_FRAMES_PER_CHUNK} frames/chunk, "
                f"{TOKENS_PER_CHUNK} tokens/chunk, embed_dim={EMBED_DIM}")

    # Load checkpoint once on CPU
    ckpt = torch.load(model_path, weights_only=True, map_location="cpu")["encoder"]
    ckpt = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt.items()}

    models = []
    for gpu_id in gpu_ids:
        logger.info(f"Loading model replica on GPU {gpu_id}...")
        model = vit_giant_xformers_rope(img_size=(IMG_SIZE, IMG_SIZE), num_frames=NUM_FRAMES_PER_CHUNK)
        model.eval()
        msg = model.load_state_dict(ckpt, strict=False)
        if gpu_id == gpu_ids[0]:
            logger.info(f"Loaded weights (msg: {msg})")
        wrapped = SingleGPUViT(model, gpu_id)
        models.append(wrapped)
        logger.info(f"Model replica ready on GPU {gpu_id}")

    return models


# ─── Inference ───────────────────────────────────────────────────────────────

def _run_batch_on_gpu(model: SingleGPUViT, batch_chunks: List[np.ndarray],
                      mean: torch.Tensor, std: torch.Tensor) -> List[np.ndarray]:
    """Run a batch of chunks through a model on its GPU. Returns list of per-chunk features."""
    tensors = []
    for chunk in batch_chunks:
        x = torch.from_numpy(chunk).float().permute(3, 0, 1, 2) / 255.0
        x = (x - mean) / std
        tensors.append(x)

    # Stack into batch: (B, 3, 64, H, W)
    x_batch = torch.stack(tensors, dim=0)
    feats_batch = model(x_batch).cpu().numpy()  # (B, tokens, embed_dim)

    # Split back into per-chunk results: each (1, tokens, embed_dim)
    return [feats_batch[i:i+1] for i in range(feats_batch.shape[0])]


def run_inference(models: List[SingleGPUViT], frames: np.ndarray) -> np.ndarray:
    """
    Run VJEPA2.1 on the entire video using data parallelism.
    Processes CHUNKS_PER_GPU chunks per GPU in parallel.
    With 4 GPUs and CHUNKS_PER_GPU=2, processes 8 chunks per batch.

    Args:
        models: List of model wrappers (one per GPU)
        frames: (T, H, W, 3) uint8

    Returns:
        np.ndarray (1, total_tokens, embed_dim) float16
    """
    T = frames.shape[0]
    num_chunks = (T + NUM_FRAMES_PER_CHUNK - 1) // NUM_FRAMES_PER_CHUNK
    num_gpus = len(models)
    total_parallel = num_gpus * CHUNKS_PER_GPU
    logger.info(f"Inference: {T} frames -> {num_chunks} chunks of {NUM_FRAMES_PER_CHUNK} | "
                f"{num_gpus} GPUs x {CHUNKS_PER_GPU} chunks/GPU = {total_parallel} parallel")

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1, 1)

    # Prepare all chunks
    chunks = []
    for i in range(num_chunks):
        start = i * NUM_FRAMES_PER_CHUNK
        end = min(start + NUM_FRAMES_PER_CHUNK, T)
        chunk = frames[start:end]

        # Pad last chunk with repeated last frame
        if chunk.shape[0] < NUM_FRAMES_PER_CHUNK:
            pad = np.repeat(chunk[-1:], NUM_FRAMES_PER_CHUNK - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        chunks.append(chunk)

    all_feats = []

    # Process chunks in batches of total_parallel (num_gpus * CHUNKS_PER_GPU)
    for batch_start in range(0, num_chunks, total_parallel):
        batch_end = min(batch_start + total_parallel, num_chunks)
        batch_chunks = chunks[batch_start:batch_end]

        # Distribute chunks across GPUs: each GPU gets up to CHUNKS_PER_GPU
        gpu_assignments = []  # list of (model, [chunk1, chunk2, ...])
        for gpu_idx in range(num_gpus):
            start_idx = gpu_idx * CHUNKS_PER_GPU
            end_idx = min(start_idx + CHUNKS_PER_GPU, len(batch_chunks))
            if start_idx < len(batch_chunks):
                gpu_chunks = batch_chunks[start_idx:end_idx]
                gpu_assignments.append((models[gpu_idx], gpu_chunks))

        active_gpus = len(gpu_assignments)
        logger.info(f"  Batch [{batch_start+1}-{batch_end}]/{num_chunks} "
                    f"({batch_end - batch_start} chunks on {active_gpus} GPUs)")

        # Run in parallel using threads (GIL released during CUDA ops)
        with ThreadPoolExecutor(max_workers=active_gpus) as executor:
            futures = []
            for model, gpu_chunks in gpu_assignments:
                future = executor.submit(_run_batch_on_gpu, model, gpu_chunks, mean, std)
                futures.append(future)

            for future in futures:
                chunk_feats_list = future.result()
                all_feats.extend(chunk_feats_list)

        logger.debug(f"  Batch done, chunk shape: {all_feats[-1].shape}")

    result = np.concatenate(all_feats, axis=1).astype(np.float16)
    logger.info(f"Final embedding: {result.shape} ({result.dtype})")
    return result
