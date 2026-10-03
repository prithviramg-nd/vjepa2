# video_classification — V-JEPA 2.1 backbone + attentive probe for Jetson

Self-contained package. Nothing outside this folder is modified; upstream V-JEPA 2.1
modules (`app/vjepa_2_1`, `src/models`) are reused by import.

## Target

| | |
|---|---|
| input clip | 6 s @ 5 fps → **30 frames × 1 (grayscale) × 224 × 224**, layout `[B, T, C, H, W]` float32 |
| tokens | tubelet 2 × patch 16 → 15 × 14 × 14 = **2940** |
| device | Jetson Orin NX 8 GB, TensorRT 10.3, 15 W mode, shared with the live analytics stack |
| budget | **FP16 TensorRT backbone ≤ 40 ms** (mean trtexec latency) |

## Selected backbone

**width (embed_dim) 192, depth 7, 3 heads (head_dim 64), MLP ratio 4 — 3.26M params.**
Output: `[B, 2940, 192]` tokens (inference); `[B, 2940, 4×192]` hierarchical (training).

Measured on Jetson Orin NX 172.16.19.193 (FP16 TRT, GPU shared with the analytics stack):

| width × depth | params | mean | median | p99 |
|---|---|---|---|---|
| **192 × 7** | **3.26M** | **38.7 ms** | 40.3 ms | 50.5 ms |
| 192 × 8 | 3.71M | 43.4 ms | 45.2 ms | 55.1 ms |
| 256 × 5 | 4.15M | 36.0 ms | 39.5 ms | 49.8 ms |
| 320 × 4 | 5.18M | 42.3 ms | 43.8 ms | 55.3 ms |
| 384 × 7 | 12.7M | 84.1 ms | 85.9 ms | 111.3 ms |

Cost per block: ~5.1 ms at width 192, ~6.8 ms at 256, ~11.6 ms at 384 — global attention
over 2940 tokens is ~45% of it (fused `_gemm_mha_v2` kernel).

## Layout

```
video_classification/
  configs/backbone_search.yaml   input / backbone / export / device / search (single source of truth)
  core/
    config.py                    typed dataclasses over the YAML (unknown keys fail loudly)
    registry.py                  name -> builder registry (backbones, heads; later datasets, transforms, loggers)
    logging_utils.py             console + file logging, silences exporter noise
  models/
    backbone.py                  VJEPA21Backbone (any depth), DeployBackbone (export wrapper), tubelet padding
    export_attention.py          TRT-friendly RoPE attention used only at export time
    classifier.py                VideoClassifier = backbone + stock AttentiveClassifier probe
  deploy/
    onnx_export.py               PyTorch -> ONNX, numerically checked against FP32 PyTorch
    trtexec.py                   trtexec command builders + log parser
    device.py                    SSH/SFTP session: upload, build engine, benchmark, save logs
  tools/
    benchmark_backbone.py        one config -> latency
    search_backbone.py           latency-constrained search for the largest backbone (resumable)
```

## Backbone

`VJEPA21Backbone` subclasses the upstream 2.1 `VisionTransformer` unchanged except for one
thing: upstream hard-codes its 4 hierarchical output levels for depth ∈ {12, 24, 40, 48}.
Here they are derived from the real depth (evenly spaced; identical to upstream for 12/24/40),
so shallow encoders work with the 2.1 hierarchical distillation loss and predictor as-is.
All other 2.1 features stay on: 3D RoPE with `interpolate_rope`, modality embeddings,
image path (`img_temporal_dim_size: 1`).

## Getting it fast on TensorRT (all fixes in PyTorch, never ONNX edits)

| step | latency (192-d, 4 blocks, 50 frames) |
|---|---|
| stock export (SDPA, FP32 graph) — TRT kept FP32/TF32 GEMMs, unfused 4900² attention, 620 MB activations | 252 ms |
| FP16 graph + `ExportRoPEAttention` | **50 ms** |

* **Exporter**: `torch.onnx.export(..., dynamo=True)`. The legacy TorchScript exporter
  mis-infers ranks in the upstream RoPE concat and aborts.
* **FP16 graph** (`export.half`): weights cast before export; I/O stays float32, casts are in-graph.
* **`ExportRoPEAttention`** (`export.explicit_attention`), swapped in by `DeployBackbone` only:
  RoPE sin/cos tables precomputed for the fixed grid and stored as buffers (upstream rebuilds
  them in FP32 every call → float/float16 mix + extra kernels), and attention written as
  `softmax(q kᵀ · scale) v`. Reuses the trained `qkv`/`proj`; matches upstream to ~1e-6.
  Training keeps upstream SDPA attention, so trained checkpoints export directly.

## Running

```bash
# one config
JETSON_PASSWORD=... ./venv/bin/python -m video_classification.tools.benchmark_backbone \
    --embed_dim 384 --num_heads 6 --depth 6

# full search (resumable: re-run the same command after an interruption)
JETSON_PASSWORD=... ./venv/bin/python -m video_classification.tools.search_backbone
```

Outputs go to `search.output_dir` (default
`/home/ubuntu/inwdata/prithvi/claude_workarea/video_classification/backbone_search`):
`results.jsonl` (one row per measurement, the resume cache), `summary.csv`, `search.log`,
`onnx/`, and full trtexec build/infer logs under `logs/`.

trtexec flags follow the device commands with two TRT 10 adjustments: `--workspace` became
`--memPoolSize=workspace:2048M`, and `--separateProfileRun` keeps `--dumpProfile`'s per-layer
timing out of the measured run. `--timingCacheFile` only speeds up repeated builds.

## Next: training (to be specified)

Planned extension points, so new stages plug in without touching existing code:

* `data/` — clip pipeline as composable, registered stages (decode → 30→5 fps resample →
  trim → crop → grayscale → normalize), driven by YAML.
  Use `models.backbone.pad_to_tubelet` so training and deployment see the same frames.
* `training/` — trainer with checkpoint/resume (`latest.pth.tar`), so a restarted machine
  continues where it stopped. A systemd unit can then provide autostart.
* tracking — MLflow behind a small logger interface (see `app/vjepa_2_1_face/mlflow_utils.py`).
