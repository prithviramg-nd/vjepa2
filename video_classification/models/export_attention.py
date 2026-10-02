"""TensorRT-friendly drop-in for upstream ``RoPEAttention`` (inference only).

Same weights and same math as ``app.vjepa_2_1.models.utils.modules.RoPEAttention``,
but for a fixed token grid:
  * the 3D RoPE sin/cos tables are computed once and stored as buffers, so they
    follow the module dtype (upstream builds them in FP32 every call, which mixes
    float/float16 in an FP16 graph and adds Sin/Cos/Slice kernels per layer);
  * attention is written as softmax(q k^T * scale) v, the pattern TensorRT fuses.
"""

import torch
import torch.nn as nn

from app.vjepa_2_1.models.utils.modules import RoPEAttention


def _rope_freqs(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Upstream ``rotate_queries_or_keys`` frequencies, repeated per (even, odd) pair: [N, dim]."""
    omega = torch.arange(dim // 2, dtype=torch.float64) / (dim / 2.0)
    omega = 1.0 / 10000**omega
    return (pos.double()[:, None] * omega[None, :]).repeat_interleave(2, dim=-1)


def _rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class ExportRoPEAttention(nn.Module):
    def __init__(self, src: RoPEAttention, grid_thw):
        super().__init__()
        if src.n_registers or src.has_cls_first:
            raise NotImplementedError("registers / cls token not supported for export")
        self.num_heads, self.head_dim, self.scale = src.num_heads, src.head_dim, src.scale
        self.qkv, self.proj = src.qkv, src.proj
        self.grid_thw = tuple(grid_thw)

        T, H, W = self.grid_thw
        ids = torch.arange(T * H * W)
        d_pos, h_pos, w_pos = src.separate_positions(ids, H, W)
        if src.interpolate_rope:
            h_pos = h_pos * (src.pretrained_grid_size - 1) / (H - 1)
            w_pos = w_pos * (src.pretrained_grid_size - 1) / (W - 1)

        # One table over the whole head: [depth | height | width | untouched] segments,
        # the untouched tail gets cos=1, sin=0 (identity), exactly like upstream's concat.
        freqs = torch.cat(
            [_rope_freqs(d_pos, src.d_dim), _rope_freqs(h_pos, src.h_dim), _rope_freqs(w_pos, src.w_dim)], dim=-1
        )
        n_rot = freqs.shape[-1]
        cos = torch.ones(T * H * W, self.head_dim, dtype=torch.float64)
        sin = torch.zeros(T * H * W, self.head_dim, dtype=torch.float64)
        cos[:, :n_rot], sin[:, :n_rot] = freqs.cos(), freqs.sin()
        self.register_buffer("rope_cos", cos.float(), persistent=False)
        self.register_buffer("rope_sin", sin.float(), persistent=False)

    def forward(self, x, mask=None, T=None, H_patches=None, W_patches=None, return_attn=False):
        if mask is not None:
            raise NotImplementedError("masked tokens are a training-only path")
        if (T, H_patches, W_patches) != self.grid_thw:
            raise ValueError(f"export attention built for grid {self.grid_thw}, got {(T, H_patches, W_patches)}")
        B, N, C = x.shape
        q, k, v = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q = q * self.rope_cos + _rotate_pairs(q) * self.rope_sin
        k = k * self.rope_cos + _rotate_pairs(k) * self.rope_sin
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x), None
