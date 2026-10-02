import logging
import os

# Libraries that log every graph-optimizer step at INFO during ONNX export.
_NOISY = ("onnxscript", "onnx_ir", "torch.onnx", "torch._export", "torch.export", "paramiko")


def setup_logging(log_file: str = None, level: int = logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
