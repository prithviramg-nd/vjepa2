import copy
from dataclasses import replace

import pytest
import torch

from video_classification.core.config import BackboneSpec, InputSpec
from video_classification.deploy.trtexec import parse_infer_log
from video_classification.models.backbone import (
    DeployBackbone,
    build_backbone,
    hierarchical_levels,
    num_tokens,
    pad_to_tubelet,
)
from video_classification.models.classifier import VideoClassifier, build_attentive_probe

SMALL = InputSpec(batch_size=2, num_frames=6, in_chans=1, height=64, width=64)
SPEC = BackboneSpec(embed_dim=96, num_heads=3, depth=4)


@pytest.mark.parametrize(
    "depth,expected", [(12, [2, 5, 8, 11]), (24, [5, 11, 17, 23]), (40, [9, 19, 29, 39]), (6, [1, 2, 4, 5])]
)
def test_hierarchical_levels_match_upstream_table(depth, expected):
    assert hierarchical_levels(depth) == expected


def test_shallow_backbone_inference_and_hierarchical_shapes():
    m = build_backbone(SPEC, SMALL).eval()
    x = torch.randn(2, 1, 6, 64, 64)
    with torch.no_grad():
        assert m(x).shape == (2, num_tokens(SPEC, SMALL), 96)
        assert m(x, training=True).shape == (2, num_tokens(SPEC, SMALL), 4 * 96)


@pytest.mark.parametrize("frames", [6, 5])  # 5 exercises tubelet padding
def test_export_attention_matches_upstream(frames):
    inp = replace(SMALL, num_frames=frames)
    torch.manual_seed(0)
    m = build_backbone(SPEC, inp).eval()
    ref, exp = DeployBackbone(copy.deepcopy(m)), DeployBackbone(m, explicit_attention=True)
    x = torch.randn(*inp.shape)
    with torch.no_grad():
        torch.testing.assert_close(exp(x), ref(x), atol=1e-5, rtol=1e-5)


def test_pad_to_tubelet_repeats_last_frame():
    x = torch.arange(5.0).view(1, 1, 5, 1, 1)
    y = pad_to_tubelet(x, 2)
    assert y.shape[2] == 6 and y[0, 0, -1].item() == 4.0 and y[0, 0, -2].item() == 4.0
    assert pad_to_tubelet(y, 2) is y


def test_classifier_forward_frozen_backbone():
    clf = VideoClassifier(build_backbone(SPEC, SMALL), build_attentive_probe(96, num_classes=3)).train()
    logits = clf(torch.randn(*SMALL.shape))
    assert logits.shape == (2, 3)
    assert not any(p.requires_grad for p in clf.encoder.parameters())
    assert not clf.encoder.training


def test_parse_trtexec_summary():
    log = (
        "[10/02/2026-07:32:12] [I] Throughput: 3.97 qps\n"
        "[10/02/2026-07:32:12] [I] Latency: min = 216.669 ms, max = 286.4 ms, mean = 251.969 ms, median = 252.851 ms, "
        "percentile(90%) = 276.468 ms, percentile(95%) = 286.4 ms, percentile(99%) = 286.4 ms\n"
        "[10/02/2026-07:32:12] [I] GPU Compute Time: min = 215.734 ms, max = 285.356 ms, mean = 250.922 ms, "
        "median = 251.714 ms, percentile(90%) = 275.455 ms, percentile(95%) = 285.356 ms, "
        "percentile(99%) = 285.356 ms\n"
        "&&&& PASSED TensorRT.trtexec [TensorRT v100300]\n"
    )
    r = parse_infer_log(log)
    assert r.passed and r.throughput_qps == 3.97
    assert r.latency.mean == 251.969 and r.gpu_compute.p99 == 285.356
