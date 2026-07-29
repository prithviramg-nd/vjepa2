# V-JEPA 2.1 — face-centered pretraining with an 8-dimensional output

Self-contained pipeline that pretrains **V-JEPA 2.1** on face-centered video clips
and produces an encoder whose per-token output width is **8** instead of the native
1024 / 1408.

Nothing in the upstream repo is modified. This app plugs in through the stock
dispatcher (`app/scaffold.py` resolves `app: vjepa_2_1_face` → `app/vjepa_2_1_face/train.py`)
and reuses `app/vjepa_2_1` for the predictor, transforms, wrappers, optimizer and
checkpoint helpers.

## Layout

```
app/vjepa_2_1_face/
  preprocess/face_crop.py        MediaPipe face detection -> median center -> clamped 512 crop
  models/vision_transformer.py   ViT subclass with the low-dim output head
  utils.py                       init_video_model (re-exports stock init_opt/load_checkpoint)
  mlflow_utils.py                fail-safe MLflow wrapper (rank 0 only)
  train.py                       trainer (copy of vjepa_2_1/train.py + low-dim + MLflow)
  scripts/run_preprocess.sh
  scripts/run_train.sh
configs/train_2_1_face/faces-8d-384px-30f.yaml
workarea/                        data, weights, checkpoints, mlruns, logs (all artifacts)
```

## How the 8-d output works

The stock 2.1 encoder routes **every** output path through `self.norms_block`:

| path | stock output | here |
|---|---|---|
| training (4 hierarchical levels concatenated) | `4 * embed_dim` | `4 * output_dim` = 32 |
| inference (`norms_block[-1]`) | `embed_dim` (1408) | `output_dim` = **8** |

So replacing each `norms_block[i]` (a `LayerNorm`) with
`Sequential(LayerNorm(D), Linear(D, 8))` is sufficient — **no `forward()` override**.

The ViT trunk keeps a real width (384 for `vit_small_lowdim`), so attention still has
capacity to learn; only the *output* is 8-d. Crucially the projection sits **inside**
the JEPA objective (the predictor is built with `embed_dim=out_embed_dim=output_dim`),
so those 8 dimensions are trained to be predictive rather than bolted on afterwards.

### Output shapes

Because of RoPE + `interpolate_rope: true`, the model trains at 384 and still runs at 768:

| input | token grid | encoder output |
|---|---|---|
| 30 x 384 x 384 | 15 x 24 x 24 | `[15, 24, 24, 8]` |
| 30 x 768 x 768 | 15 x 48 x 48 | `[15, 48, 48, 8]` |

## Preprocessing

1. MediaPipe (`blaze_face_short_range.tflite`) runs on **every** frame.
2. The **median** face center across frames is taken (robust to per-frame misses/jitter);
   the largest detection per frame is used so background faces are ignored.
3. A 512x512 window is cropped around that center, **clamped** to the frame. If the face
   is near an edge the window snaps inward and the face is intentionally off-center —
   e.g. center 900, crop 512, frame 1000 → `900-256 = 644`, `644+512 = 1156 > 1000`,
   so `x0 = 1000-512 = 488`.
4. The centre 30 frames are written at 10 FPS (short clips are loop-padded).

Measured on the 334 source clips: **100% processed, 96.8% mean per-frame detection rate,
3 videos with no face at all (fall back to frame center), 40 videos clamped to an edge.**

Outputs: `workarea/faces_512/*.mp4`, `workarea/centers.json` (audit trail),
`workarea/train.csv` (the space-delimited `<path> <label>` format the loader expects).

## Model sizes

The **predictor is discarded after training** — it exists only to compute the JEPA
loss. The artifact you actually deploy is the encoder.

| `model_name` | trunk | pred depth/dim | encoder (kept) | predictor | total trained |
|---|---|---|---|---|---|
| `vit_tiny_lowdim` | 192 | 8 / 192 | 5.6M | 3.6M | 9.2M |
| `vit_tiny_lowdim` | 192 | 12 / 192 | 5.6M | 5.4M | 11.0M |
| **`vit_small_lowdim` (default)** | 384 | 12 / 384 | **21.9M** | 21.3M | 43.2M |
| `vit_base_lowdim` | 768 | 12 / 384 | 86.3M | 21.3M | 107.6M |

All emit `output_dim`-wide tokens regardless of trunk width. Add new sizes in
`models/vision_transformer.py`; depth must stay in {12, 24, 40, 48} because the
parent class derives its hierarchical layer indices from it.

## Scaling notes (100k source videos -> ~2M clips)

100k x 1 min segmented into 3s clips = **~2M training clips**.
Measured throughput with the 21.9M encoder on 4x T4:

| config | videos/s (4 GPUs) | one pass over 2M clips |
|---|---|---|
| batch 8, crop 384 | 2.1 | **~11 days** |
| batch 16, crop 256 | 7.9 | **~2.9 days** |

| | 334 clips (measured) | 2M clips (extrapolated) |
|---|---|---|
| preprocessing | 22s @ 8 workers | ~6 h @ 48 cores |
| storage | 46 MB (138 KB/clip) | ~276 GB |

Preprocessing and storage scale fine; **training is the bottleneck**. Even at crop 256
a single pass is ~3 days on 4x T4, and SSL pretraining wants many passes. Consider
subsampling the corpus, or larger/more GPUs.

## Running

```bash
./app/vjepa_2_1_face/scripts/run_preprocess.sh    # step 1
./app/vjepa_2_1_face/scripts/run_train.sh         # step 2 (4x T4)
mlflow ui --backend-store-uri sqlite:///workarea/mlflow.db   # tracking
```

Training progress also lands in `workarea/runs/<run>/log_r*.csv` and
`workarea/logs/train4gpu.log`; checkpoints are written every epoch to
`latest.pth.tar`, so the run can be stopped at any point and resumed
(`meta.load_checkpoint: true`).

## Extracting 8-d features

```bash
python -m app.vjepa_2_1_face.extract_features \
    --checkpoint workarea/runs/faces-8d-384px-30f/latest.pth.tar \
    --videos    /home/ubuntu/inwdata/prithvi/videos \
    --resolution 768 --device cuda:0 --pooled \
    --out workarea/features.npz
```

Applies the identical face-crop preprocessing, then the EMA `target_encoder` in
inference mode. Verified output: `(15, 48, 48, 8)` at 768 and `(15, 24, 24, 8)` at 384
from the same checkpoint. `--pooled` additionally stores one mean-pooled 8-d vector per
video (handy for retrieval/clustering).

Note: 768 inference is ~34.5k tokens, so point `--device` at a GPU that is not busy
training, or use `--device cpu`. A harmless `TypeError: 'NoneType' object is not callable`
may print at exit — that is a known MediaPipe 0.10.35 destructor bug, after all work is done.

## Hardware notes (4x T4, 16GB)

These were found empirically, not assumed:

* **`dtype: float16`** — T4 is Turing and has no native bfloat16, so the stock
  `bfloat16` config would be emulated and slow.
* **`NCCL_P2P_DISABLE=1` + `NCCL_SHM_DISABLE=1`** — these T4s report no peer access for
  any pair (`Cuda failure 217`). `P2P_DISABLE` alone is **not** enough; both are needed
  to push NCCL onto socket transport. Cheap here since the model is only ~43M params.
* **Batch size / resolution** — measured on one idle T4 with the 21.9M encoder
  (`scripts/bench.sh`, 12 iterations per config, 30 frames):

  | batch/GPU | crop | peak mem | s/iter | videos/s/GPU |
  |---|---|---|---|---|
  | 8 | 384 | 6.07 GB | 15.03 | 0.53 |
  | 16 | 384 | 11.13 GB | 27.32 | 0.59 |
  | 24 | 384 | **OOM** | - | - |
  | 16 | 256 | 5.21 GB | 8.07 | **1.98** |
  | 32 | 256 | 9.47 GB | 15.77 | 2.03 |

  Two conclusions: the T4 is **compute-bound**, so raising the batch at a fixed
  resolution buys almost nothing (8→16 at 384 is only +11% for ~2x the memory);
  and **resolution is the real lever** — 384→256 is a **3.7x** speedup.
  Recommended: **batch 16 at crop 256** (fast, and 5.2GB leaves room for DDP).
  Activation checkpointing is required in all cases.
* Augmentation uses `random_resize_scale: [0.7, 1.0]` instead of the stock `[0.3, 1.0]`:
  the clips are deliberately face-centered and an aggressive random crop would throw the
  face back out of frame.

## Caveat on data volume

334 clips (~17 minutes) is far below what JEPA pretraining from scratch needs; expect the
loss to fall without the features becoming broadly useful. The pipeline is built to scale —
point `--videos` at a larger folder, rerun preprocessing, and retrain.
