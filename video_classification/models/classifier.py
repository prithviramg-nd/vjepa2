"""Backbone + attentive probe video classifier.

The probe is the stock V-JEPA ``AttentiveClassifier`` (cross-attention pooling
from learned queries, then a linear layer); nothing here re-implements it.
"""

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentiveClassifier
from video_classification.core.registry import Registry
from video_classification.models.backbone import DeployBackbone, VJEPA21Backbone

HEADS = Registry("head")


@HEADS.register("attentive_probe")
def build_attentive_probe(embed_dim: int, num_classes: int, num_heads: int = None, depth: int = 1, **kwargs):
    return AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads or max(1, embed_dim // 64),
        depth=depth,
        num_classes=num_classes,
        **kwargs,
    )


class VideoClassifier(nn.Module):
    def __init__(self, backbone: VJEPA21Backbone, head: nn.Module, freeze_backbone: bool = True):
        super().__init__()
        self.encoder = DeployBackbone(backbone)
        self.head = head
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: [B, T, C, H, W] -> logits [B, num_classes]."""
        with torch.set_grad_enabled(self.training and not self.freeze_backbone):
            feats = self.encoder(clip)
        return self.head(feats)
