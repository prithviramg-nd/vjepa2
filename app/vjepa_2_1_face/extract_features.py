"""Extract 8-dimensional V-JEPA 2.1 features from videos.

Runs the same face-centered preprocessing used for training, then the trained
encoder in inference mode, producing per-token features of width `output_dim`.

    [T // tubelet_size, H // patch, W // patch, output_dim]

Because the encoder uses RoPE with `interpolate_rope`, a model trained at 384 can
be evaluated at 768:

    --resolution 384  ->  [15, 24, 24, 8]
    --resolution 768  ->  [15, 48, 48, 8]

Example:
    python -m app.vjepa_2_1_face.extract_features \
        --checkpoint workarea/runs/faces-8d-384px-30f/latest.pth.tar \
        --videos /home/ubuntu/inwdata/prithvi/videos \
        --resolution 768 --out workarea/features.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.vjepa_2_1_face.preprocess.face_crop import (
    clamp_window,
    detect_median_center,
    pad_to_min,
    read_frames,
)
from app.vjepa_2_1_face.utils import init_video_model

# must match app/vjepa_2_1/transforms.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def build_encoder(checkpoint, params, device, which="target_encoder"):
    m = params["model"]
    d = params["data"]
    encoder, _ = init_video_model(
        device=device,
        patch_size=d["patch_size"],
        max_num_frames=max(d["dataset_fpcs"]),
        tubelet_size=d["tubelet_size"],
        model_name=m["model_name"],
        output_dim=m.get("output_dim", 8),
        head_hidden_ratio=m.get("head_hidden_ratio", 0.0),
        crop_size=d["crop_size"],
        pred_depth=m["pred_depth"],
        pred_num_heads=m.get("pred_num_heads"),
        pred_embed_dim=m["pred_embed_dim"],
        uniform_power=m.get("uniform_power", False),
        use_mask_tokens=m.get("use_mask_tokens", False),
        num_mask_tokens=int(len(params["mask"]) * len(d["dataset_fpcs"])),
        zero_init_mask_tokens=m.get("zero_init_mask_tokens", True),
        use_sdpa=params["meta"].get("use_sdpa", True),
        use_rope=m.get("use_rope", True),
        use_activation_checkpointing=False,
        return_all_tokens=params["loss"].get("predict_all", True),
        img_temporal_dim_size=m.get("img_temporal_dim_size"),
        n_registers=m.get("n_registers", 0),
        n_registers_predictor=m.get("n_registers_predictor", 0),
        has_cls_first=m.get("has_cls_first", False),
        interpolate_rope=m.get("interpolate_rope", True),
        modality_embedding=m.get("modality_embedding", False),
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt[which if which in ckpt else "encoder"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}  # strip DDP prefix
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    print(f"loaded '{which}' from epoch {ckpt.get('epoch')} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    encoder.eval()
    return encoder.backbone


@torch.no_grad()
def video_to_tensor(path, detector, mp_mod, crop, num_frames, resolution, device):
    """Face-crop a video and return a normalized (1, 3, T, R, R) tensor."""
    frames = read_frames(path)
    if not frames:
        return None
    cx, cy, _, _ = detect_median_center(frames, detector, mp_mod)
    frames = [pad_to_min(f, crop) for f in frames]
    h, w = frames[0].shape[:2]
    x0, y0 = clamp_window(cx, crop, w), clamp_window(cy, crop, h)

    if len(frames) >= num_frames:
        s = (len(frames) - num_frames) // 2
        sel = frames[s : s + num_frames]
    else:
        sel = [frames[i % len(frames)] for i in range(num_frames)]

    clip = np.stack([f[y0 : y0 + crop, x0 : x0 + crop, ::-1] for f in sel])  # BGR->RGB
    clip = (clip.astype(np.float32) - MEAN) / STD
    t = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # 1,C,T,H,W
    if crop != resolution:
        t = F.interpolate(
            t, size=(t.shape[2], resolution, resolution), mode="trilinear", align_corners=False
        )
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--params", default=None, help="params-pretrain.yaml (default: next to checkpoint)")
    ap.add_argument("--videos", required=True, help="video file or directory")
    ap.add_argument("--out", default="/home/ubuntu/inwdata/prithvi/git/vjepa2/workarea/features.npz")
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--which", default="target_encoder", choices=["target_encoder", "encoder"])
    ap.add_argument("--pooled", action="store_true", help="also store the mean-pooled 8-d vector")
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="cuda:N or cpu. Note: 768-res inference is ~34.5k tokens, so pick a GPU "
        "that is not busy training, or use cpu.",
    )
    ap.add_argument(
        "--face-model",
        default="/home/ubuntu/inwdata/prithvi/git/vjepa2/workarea/models/blaze_face_short_range.tflite",
    )
    args = ap.parse_args()

    params_path = args.params or os.path.join(os.path.dirname(args.checkpoint), "params-pretrain.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    backbone = build_encoder(args.checkpoint, params, device, args.which)

    import mediapipe as mp_mod
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=args.face_model),
            min_detection_confidence=0.4,
        )
    )

    if os.path.isdir(args.videos):
        vids = sorted(
            os.path.join(args.videos, f)
            for f in os.listdir(args.videos)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
        )
    else:
        vids = [args.videos]

    tub = params["data"]["tubelet_size"]
    patch = params["data"]["patch_size"]
    grid = args.resolution // patch
    feats, pooled, names = {}, {}, []

    from tqdm import tqdm

    for p in tqdm(vids, desc="extract"):
        t = video_to_tensor(p, detector, mp_mod, args.crop, args.num_frames, args.resolution, device)
        if t is None:
            continue
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            z = backbone(t, training=False)  # (1, N, output_dim)
        z = z.float().squeeze(0).detach()
        z = z.reshape(args.num_frames // tub, grid, grid, z.shape[-1])  # [T,H,W,D]
        key = os.path.splitext(os.path.basename(p))[0]
        feats[key] = z.cpu().numpy().astype(np.float32)
        if args.pooled:
            pooled[key] = z.mean(dim=(0, 1, 2)).cpu().numpy().astype(np.float32)
        names.append(key)

    shape = next(iter(feats.values())).shape if feats else None
    print(f"extracted {len(feats)} videos, per-video feature shape {shape}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.pooled:
        np.savez_compressed(args.out, **{f"feat/{k}": v for k, v in feats.items()},
                            **{f"pool/{k}": v for k, v in pooled.items()})
    else:
        np.savez_compressed(args.out, **feats)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
