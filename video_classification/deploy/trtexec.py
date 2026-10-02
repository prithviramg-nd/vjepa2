"""trtexec command construction and output parsing."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


def build_cmd(trtexec: str, onnx_path: str, engine_path: str, flags: List[str]) -> str:
    return " ".join([trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", *flags])


def infer_cmd(trtexec: str, engine_path: str, flags: List[str]) -> str:
    return " ".join([trtexec, f"--loadEngine={engine_path}", *flags])


_STAT = r"min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms"
_KEYS = ("min", "max", "mean", "median", "p90", "p95", "p99")


@dataclass
class LatencyStats:
    min: float
    max: float
    mean: float
    median: float
    p90: float
    p95: float
    p99: float


@dataclass
class TrtResult:
    latency: Optional[LatencyStats]  # host-to-host, includes H2D/D2H copies
    gpu_compute: Optional[LatencyStats]  # kernel time only
    throughput_qps: Optional[float]
    passed: bool

    def as_flat_dict(self) -> Dict[str, float]:
        out = {"throughput_qps": self.throughput_qps}
        for prefix, stats in (("lat", self.latency), ("gpu", self.gpu_compute)):
            for k in _KEYS:
                out[f"{prefix}_{k}"] = getattr(stats, k) if stats else None
        return out


def _parse_stats(log: str, label: str) -> Optional[LatencyStats]:
    # Performance summary lines look like: "[I] Latency: min = ... ms, ..., percentile(99%) = ... ms"
    m = re.search(rf"\] {label}: {_STAT}", log)
    return LatencyStats(*map(float, m.groups())) if m else None


def parse_infer_log(log: str) -> TrtResult:
    tput = re.search(r"Throughput: ([\d.]+) qps", log)
    return TrtResult(
        latency=_parse_stats(log, "Latency"),
        gpu_compute=_parse_stats(log, "GPU Compute Time"),
        throughput_qps=float(tput.group(1)) if tput else None,
        passed="PASSED TensorRT.trtexec" in log,
    )
