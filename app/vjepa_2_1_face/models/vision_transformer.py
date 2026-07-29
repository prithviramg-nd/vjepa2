"""V-JEPA 2.1 encoder with a low-dimensional output head.

Goal: make the encoder emit `output_dim` (e.g. 8) channels instead of the native
backbone width (1024 / 1408 / ...), while keeping the full-width ViT trunk so the
model still has the capacity to learn.

Implementation note -- why there is no `forward()` override here:

The stock `VisionTransformer.forward` funnels *every* output path through
`self.norms_block`:

    out_norm = self.norms_block[out_idx](x)      # out_layers path
    hier.append(self.norms_block[out_idx](x))    # hierarchical distillation targets (training)
    x = self.norms_block[-1](x)                  # final output (inference)

So replacing each `norms_block[i]` (a LayerNorm) with
`Sequential(LayerNorm(D), Linear(D, output_dim))` gives us, for free:

    training  : cat of 4 levels -> 4 * output_dim   (e.g. 32) -- what the JEPA loss sees
    inference : output_dim                          (e.g. 8)  -- [T/tubelet, H/p, W/p, 8]

The trunk, attention, RoPE and every other behaviour are untouched, and the
projection sits inside the JEPA objective so those 8 dims are genuinely trained
to be predictive rather than being a bolted-on afterthought.
"""

from functools import partial

import torch.nn as nn
from src.utils.tensors import trunc_normal_

from app.vjepa_2_1.models.vision_transformer import VisionTransformer


class ProjectionHead(nn.Module):
    """LayerNorm(D) -> [optional hidden MLP] -> Linear(D, output_dim)."""

    def __init__(self, norm: nn.Module, embed_dim: int, output_dim: int, hidden_ratio: float = 0.0):
        super().__init__()
        self.norm = norm
        if hidden_ratio and hidden_ratio > 0:
            hidden = max(output_dim, int(embed_dim * hidden_ratio))
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, hidden, bias=True),
                nn.GELU(),
                nn.Linear(hidden, output_dim, bias=True),
            )
        else:
            self.proj = nn.Linear(embed_dim, output_dim, bias=True)

    def forward(self, x):
        return self.proj(self.norm(x))


class VisionTransformerLowDim(VisionTransformer):
    """VisionTransformer whose per-token output width is `output_dim`."""

    def __init__(self, output_dim=8, head_hidden_ratio=0.0, init_std=0.02, **kwargs):
        super().__init__(init_std=init_std, **kwargs)
        self.output_dim = output_dim
        self.trunk_embed_dim = self.embed_dim

        # Wrap each hierarchical-level norm with a projection to `output_dim`.
        # Each level gets its own head so the 4 levels stay distinct.
        self.norms_block = nn.ModuleList(
            [
                ProjectionHead(norm, self.trunk_embed_dim, output_dim, head_hidden_ratio)
                for norm in self.norms_block
            ]
        )

        for m in self.norms_block.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def _build(depth, num_heads, embed_dim, patch_size=16, **kwargs):
    return VisionTransformerLowDim(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


# Trunk widths mirror the stock V-JEPA 2.1 variants; only the OUTPUT width changes.
# Depth must stay in {12, 24, 40, 48} -- the parent picks hierarchical layers from it.
def vit_tiny_lowdim(patch_size=16, **kwargs):
    return _build(12, 3, 192, patch_size, **kwargs)


def vit_small_lowdim(patch_size=16, **kwargs):
    return _build(12, 6, 384, patch_size, **kwargs)


def vit_base_lowdim(patch_size=16, **kwargs):
    return _build(12, 12, 768, patch_size, **kwargs)


def vit_large_lowdim(patch_size=16, **kwargs):
    return _build(24, 16, 1024, patch_size, **kwargs)
