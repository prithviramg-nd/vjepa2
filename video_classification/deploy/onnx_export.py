"""PyTorch -> ONNX export (fixed shapes, numerically checked).

Uses the torch.export-based exporter: the legacy TorchScript exporter
mis-infers tensor ranks in the V-JEPA 2.1 RoPE concat and aborts.

Policy: if TensorRT cannot convert a layer, fix the PyTorch module and
re-export. The ONNX graph is never edited by hand or rewritten afterwards.
"""

import logging

import numpy as np
import onnx
import torch

logger = logging.getLogger(__name__)

HALF_REL_TOL = 2e-2


def export_onnx(
    model: torch.nn.Module,
    input_shape,
    path: str,
    half: bool = False,
    opset: int = 18,
    check_numerics: bool = True,
    output_name: str = "features",
    atol: float = 1e-3,
) -> str:
    """Export ``model`` (input named after ``forward``'s first arg).

    With ``half`` the weights are cast to FP16 for export; the ONNX output is
    then checked against the FP32 model with a relative tolerance.
    """
    model = model.eval().cpu().float()
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        ref = model(dummy).numpy() if check_numerics else None
        if half:
            model = model.half()
        torch.onnx.export(
            model,
            (dummy,),
            path,
            output_names=[output_name],
            opset_version=opset,
            dynamo=True,
            optimize=True,
            external_data=False,
        )
    onnx.checker.check_model(path)

    if check_numerics:
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]
        err = float(np.abs(got - ref).max())
        rel = err / float(np.abs(ref).max())
        if (half and rel > HALF_REL_TOL) or (not half and err > atol):
            raise RuntimeError(f"ONNX output differs from PyTorch FP32: max abs err {err:.2e} (rel {rel:.2e})")
        logger.info("  ONNX numerics OK vs FP32 PyTorch (max abs err %.2e, rel %.2e)", err, rel)

    return path
