"""V-JEPA 2.1 encoder with a free choice of depth.

The upstream ``app.vjepa_2_1.models.vision_transformer.VisionTransformer``
hard-codes its 4 hierarchical output levels for depth in {12, 24, 40, 48},
which rules out the shallow encoders a 25 ms Jetson budget calls for. This
subclass derives the levels from the actual depth (evenly spaced, matching the
upstream table for 12/24/40) and otherwise reuses the upstream class unchanged,
so V-JEPA 2.1 pretraining code (predictor, hierarchical distillation loss) works
on it as-is.
"""

import contextlib
import io
from dataclasses import replace
from functools import partial

import torch
import torch.nn as nn

from app.vjepa_2_1.models.vision_transformer import VisionTransformer
from video_classification.core.config import BackboneSpec, InputSpec
from video_classification.core.registry import Registry
from video_classification.models.export_attention import ExportRoPEAttention

BACKBONES = Registry("backbone")

N_LEVELS = 4


def hierarchical_levels(depth: int, n_levels: int = N_LEVELS):
    """Block indices whose outputs form the hierarchical feature levels."""
    if depth < n_levels:
        raise ValueError(f"depth must be >= {n_levels}, got {depth}")
    # integer ceil, not round(): Python's banker's rounding skews odd splits (depth 6 -> [1, 2, 3, 5])
    return [-(-(i + 1) * depth // n_levels) - 1 for i in range(n_levels)]


class VJEPA21Backbone(VisionTransformer):
    # The parent assigns these from its fixed depth table (and skips them for
    # other depths); derive them from the real block count instead.
    @property
    def hierarchical_layers(self):
        return hierarchical_levels(len(self.blocks))

    @hierarchical_layers.setter
    def hierarchical_layers(self, _):
        pass

    @property
    def out_layers_distillation(self):
        return self.hierarchical_layers

    @out_layers_distillation.setter
    def out_layers_distillation(self, _):
        pass

    @classmethod
    def from_spec(cls, spec: BackboneSpec, inp: InputSpec) -> "VJEPA21Backbone":
        if spec.embed_dim % spec.num_heads:
            raise ValueError(f"embed_dim {spec.embed_dim} not divisible by num_heads {spec.num_heads}")
        # Upstream prints "Check the code! ;)" for depths outside its table; the
        # properties above make those depths valid, so drop the message.
        with contextlib.redirect_stdout(io.StringIO()):
            return cls(**cls._kwargs(spec, inp))

    @staticmethod
    def _kwargs(spec: BackboneSpec, inp: InputSpec) -> dict:
        return dict(
            img_size=(inp.height, inp.width),
            num_frames=padded_frames(inp.num_frames, spec.tubelet_size),
            in_chans=inp.in_chans,
            patch_size=spec.patch_size,
            tubelet_size=spec.tubelet_size,
            embed_dim=spec.embed_dim,
            depth=spec.depth,
            num_heads=spec.num_heads,
            mlp_ratio=spec.mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_rope=spec.use_rope,
            interpolate_rope=spec.interpolate_rope,
            modality_embedding=spec.modality_embedding,
            img_temporal_dim_size=spec.img_temporal_dim_size,
            use_silu=spec.use_silu,
            drop_path_rate=spec.drop_path_rate,
            use_activation_checkpointing=spec.use_activation_checkpointing,
            n_output_distillation=N_LEVELS,
        )


class DeployBackbone(nn.Module):
    """Inference wrapper: [B, T, C, H, W] float32 clip -> [B, N_tokens, embed_dim] float32 features.

    The deployed input layout is frames-first (how frames come off the camera);
    the encoder wants channels-first, so the transpose lives in the graph, as do
    the tubelet padding and the casts when the weights are half precision.
    """

    def __init__(self, backbone: VJEPA21Backbone, explicit_attention: bool = False):
        super().__init__()
        self.backbone = backbone
        if explicit_attention:
            grid = (
                backbone.num_frames // backbone.tubelet_size,
                backbone.img_height // backbone.patch_size,
                backbone.img_width // backbone.patch_size,
            )
            for blk in backbone.blocks:
                blk.attn = ExportRoPEAttention(blk.attn, grid)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        dtype = self.backbone.patch_embed.proj.weight.dtype
        x = pad_to_tubelet(clip.transpose(1, 2), self.backbone.tubelet_size)
        return self.backbone(x.to(dtype)).float()


def padded_frames(num_frames: int, tubelet_size: int) -> int:
    return -(-num_frames // tubelet_size) * tubelet_size


def pad_to_tubelet(x: torch.Tensor, tubelet_size: int) -> torch.Tensor:
    """[B, C, T, H, W]: repeat the last frame until T is a multiple of the tubelet.

    E.g. 5 s @ 5 fps = 25 frames -> 26, so no frame is dropped by the tubelet conv.
    Training and deployment must both go through this so they see the same frames.
    """
    pad = padded_frames(x.shape[2], tubelet_size) - x.shape[2]
    if pad == 0:
        return x
    return torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)


@BACKBONES.register("vjepa21")
def build_backbone(spec: BackboneSpec, inp: InputSpec, **overrides) -> VJEPA21Backbone:
    return VJEPA21Backbone.from_spec(replace(spec, **overrides), inp)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def num_tokens(spec: BackboneSpec, inp: InputSpec) -> int:
    return (
        (padded_frames(inp.num_frames, spec.tubelet_size) // spec.tubelet_size)
        * (inp.height // spec.patch_size)
        * (inp.width // spec.patch_size)
    )
