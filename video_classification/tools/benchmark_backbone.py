"""Export one backbone config to ONNX and measure its TensorRT latency on the Jetson.

    JETSON_PASSWORD=... python -m video_classification.tools.benchmark_backbone \
        --config video_classification/configs/backbone_search.yaml \
        --embed_dim 384 --num_heads 6 --depth 7

    # another device: its own host + password env var
    JETSON2_PASSWORD=... python -m video_classification.tools.benchmark_backbone \
        --embed_dim 384 --num_heads 6 --depth 7 --host 172.16.25.140 --password_env JETSON2_PASSWORD
"""

import argparse
import json
import logging
import os
import time
from dataclasses import replace

from video_classification.core.config import load_search_config, to_dict
from video_classification.core.logging_utils import setup_logging
from video_classification.deploy.device import JetsonDevice
from video_classification.deploy.onnx_export import export_onnx
from video_classification.models.backbone import DeployBackbone, build_backbone, count_params, num_tokens

logger = logging.getLogger(__name__)


def run_name(spec, inp, exp) -> str:
    """Unique id of one measurement: architecture + input frames + export variant."""
    return f"{spec.name}_f{inp.num_frames}_{'fp16' if exp.half else 'fp32'}_{'xattn' if exp.explicit_attention else 'sdpa'}"


def export_spec(spec, inp, exp, out_dir: str) -> dict:
    """Build + export one spec; returns its metadata (incl. local onnx path)."""
    os.makedirs(out_dir, exist_ok=True)
    model = build_backbone(spec, inp)
    name = run_name(spec, inp, exp)
    onnx_path = os.path.join(out_dir, name + ".onnx")
    t0 = time.time()
    logger.info("[%s] exporting ONNX (%.1fM params, input %s)", name, count_params(model) / 1e6, list(inp.shape))
    export_onnx(
        DeployBackbone(model, explicit_attention=exp.explicit_attention),
        inp.shape,
        onnx_path,
        half=exp.half,
        opset=exp.opset,
        atol=exp.atol,
    )
    logger.info("  export done in %.0fs", time.time() - t0)
    return {
        "name": name,
        "num_frames": inp.num_frames,
        "params_m": round(count_params(model) / 1e6, 2),
        "tokens": num_tokens(spec, inp),
        "onnx": onnx_path,
        **{k: v for k, v in to_dict(spec).items() if k in ("embed_dim", "depth", "num_heads", "tubelet_size", "patch_size")},
    }


def benchmark_spec(spec, inp, exp, device: JetsonDevice, out_dir: str) -> dict:
    meta = export_spec(spec, inp, exp, os.path.join(out_dir, "onnx"))
    res = device.benchmark_onnx(meta["onnx"], os.path.join(out_dir, "logs"))
    row = {**meta, "half": exp.half, "explicit_attention": exp.explicit_attention, **res.as_flat_dict(), "trt_passed": res.passed}
    logger.info("%s  params=%.1fM  mean=%.2fms  p99=%.2fms", meta["name"], row["params_m"], row["lat_mean"] or -1, row["lat_p99"] or -1)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="video_classification/configs/backbone_search.yaml")
    p.add_argument("--embed_dim", type=int)
    p.add_argument("--num_heads", type=int)
    p.add_argument("--depth", type=int)
    p.add_argument("--num_frames", type=int, help="override input.num_frames (50 = 10 fps, 25 = 5 fps)")
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--explicit_attention", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--host", help="override device.host (e.g. a second Jetson)")
    p.add_argument("--password_env", help="env var holding that device's SSH password")
    p.add_argument("--out_dir", help="default: <search.output_dir>/devices/<host>")
    args = p.parse_args()
    inp, spec, exp, dev_cfg, search = load_search_config(args.config)
    spec = replace(spec, **{k: getattr(args, k) for k in ("embed_dim", "num_heads", "depth") if getattr(args, k) is not None})
    if args.num_frames:
        inp = replace(inp, num_frames=args.num_frames)
    exp = replace(exp, **{k: getattr(args, k) for k in ("half", "explicit_attention") if getattr(args, k) is not None})
    dev_cfg = replace(dev_cfg, **{k: getattr(args, k) for k in ("host", "password_env") if getattr(args, k)})
    out_dir = args.out_dir or os.path.join(search.output_dir, "devices", dev_cfg.host)
    setup_logging(os.path.join(out_dir, "benchmark.log"))

    with JetsonDevice(dev_cfg) as dev:
        row = benchmark_spec(spec, inp, exp, dev, out_dir)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
