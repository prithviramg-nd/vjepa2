"""Typed config objects shared by export, benchmarking and (later) training.

YAML is the source of truth; these dataclasses give it a schema so typos fail
loudly instead of being silently ignored.
"""

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


def _from_dict(cls, d: Optional[Dict[str, Any]]):
    d = d or {}
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**d)


@dataclass
class InputSpec:
    """Clip tensor fed to the backbone, laid out as [B, T, C, H, W]."""

    batch_size: int = 1
    num_frames: int = 50
    in_chans: int = 1
    height: int = 224
    width: int = 224

    @property
    def shape(self):
        return (self.batch_size, self.num_frames, self.in_chans, self.height, self.width)


@dataclass
class BackboneSpec:
    """Architecture of a V-JEPA 2.1 style ViT encoder."""

    embed_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    patch_size: int = 16
    tubelet_size: int = 2
    use_rope: bool = True
    interpolate_rope: bool = True
    modality_embedding: bool = True
    img_temporal_dim_size: Optional[int] = 1
    use_silu: bool = False
    drop_path_rate: float = 0.0
    use_activation_checkpointing: bool = False

    @property
    def name(self) -> str:
        return f"vjepa21_d{self.embed_dim}_L{self.depth}_h{self.num_heads}_t{self.tubelet_size}_p{self.patch_size}"


@dataclass
class ExportConfig:
    """How the PyTorch backbone is turned into the ONNX handed to TensorRT."""

    half: bool = True  # FP16 weights in the graph (I/O stays float32)
    explicit_attention: bool = True  # canonical matmul-softmax-matmul instead of SDPA
    opset: int = 18
    atol: float = 1e-3  # ONNX-vs-PyTorch check; relaxed automatically for half


@dataclass
class DeviceConfig:
    """Remote Jetson used for TensorRT latency measurement.

    The password is never stored in YAML; it is read from ``password_env``.
    """

    host: str = "172.16.19.193"
    user: str = "ubuntu"
    password_env: str = "JETSON_PASSWORD"
    remote_dir: str = "/home/ubuntu/latency_analysis"
    trtexec: str = "/usr/src/tensorrt/bin/trtexec"
    build_flags: List[str] = field(
        default_factory=lambda: [
            "--fp16",
            "--tacticSources=-CUBLAS_LT,-CUBLAS,-CUDNN",
            "--memPoolSize=workspace:2048M",
        ]
    )
    infer_flags: List[str] = field(
        default_factory=lambda: [
            "--warmUp=1000",
            "--iterations=1000",
            "--verbose",
            "--dumpProfile",
            "--separateProfileRun",
        ]
    )
    keep_remote_files: bool = False

    @property
    def password(self) -> str:
        pw = os.environ.get(self.password_env)
        if not pw:
            raise RuntimeError(f"Set ${self.password_env} to the device SSH password")
        return pw


@dataclass
class SearchConfig:
    """Latency-constrained architecture search."""

    budget_ms: float = 25.0
    metric: str = "mean"  # which trtexec latency statistic must stay under budget: mean|median|p95|p99
    widths: List[Dict[str, int]] = field(default_factory=list)  # [{embed_dim, num_heads}, ...]
    probe_depths: List[int] = field(default_factory=lambda: [4, 8])
    min_depth: int = 4
    max_depth: int = 24
    output_dir: str = "/home/ubuntu/inwdata/prithvi/claude_workarea/video_classification/backbone_search"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_search_config(path: str):
    raw = load_yaml(path)
    return (
        _from_dict(InputSpec, raw.get("input")),
        _from_dict(BackboneSpec, raw.get("backbone")),
        _from_dict(ExportConfig, raw.get("export")),
        _from_dict(DeviceConfig, raw.get("device")),
        _from_dict(SearchConfig, raw.get("search")),
    )


def to_dict(obj) -> Dict[str, Any]:
    return asdict(obj)
