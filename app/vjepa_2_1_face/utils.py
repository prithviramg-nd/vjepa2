"""Model construction for the low-dimensional (8-d output) V-JEPA 2.1 variant.

Only `init_video_model` differs from `app.vjepa_2_1.utils`; the optimizer and
checkpoint helpers are re-exported unchanged so there is a single source of truth.
"""

import logging
import sys

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1_face.models.vision_transformer as video_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

# re-exported unchanged from the stock 2.1 app
from app.vjepa_2_1.utils import init_opt, load_checkpoint, normalize_nested  # noqa: F401

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# number of hierarchical levels the encoder concatenates during training
N_LEVELS = 4


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_small_lowdim",
    output_dim=8,
    head_hidden_ratio=0.0,
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        output_dim=output_dim,
        head_hidden_ratio=head_hidden_ratio,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    trunk_dim = encoder.trunk_embed_dim
    encoder = MultiSeqWrapper(encoder)

    # The predictor consumes the encoder's TRAINING output (N_LEVELS * output_dim)
    # and must emit targets of the same width, hence embed_dim/out_embed_dim = output_dim.
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=output_dim,
        out_embed_dim=output_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=pred_num_heads if pred_num_heads is not None else 12,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"[lowdim] model={model_name} trunk_dim={trunk_dim} output_dim={output_dim} "
        f"(train-time encoder width = {N_LEVELS} x {output_dim} = {N_LEVELS * output_dim})"
    )
    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor
