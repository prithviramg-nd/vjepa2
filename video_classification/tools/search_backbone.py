"""Find the largest V-JEPA 2.1 backbone whose TensorRT latency fits the budget.

For each width (embed_dim, num_heads) in the config:
  1. measure the probe depths (e.g. 4 and 8) on the Jetson,
  2. fit latency = a + b * depth (each block costs the same, so this is ~linear),
  3. jump to the predicted max depth and walk +-1 until it is the deepest that fits.

Every measurement is appended to results.jsonl and reused on restart, so an
interrupted search (reboot, dropped SSH) resumes where it stopped.

    JETSON_PASSWORD=... python -m video_classification.tools.search_backbone \
        --config video_classification/configs/backbone_search.yaml
"""

import argparse
import json
import logging
import math
import os
from dataclasses import replace

import pandas as pd
from tqdm import tqdm

from video_classification.core.config import load_search_config
from video_classification.core.logging_utils import setup_logging
from video_classification.deploy.device import JetsonDevice, RemoteCommandError
from video_classification.tools.benchmark_backbone import benchmark_spec, run_name

logger = logging.getLogger(__name__)


class ResultStore:
    """Append-only JSONL cache of measurements keyed by spec name."""

    def __init__(self, path: str):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    self.rows[row["name"]] = row
            logger.info("resuming: %d cached measurements in %s", len(self.rows), path)

    def add(self, row: dict):
        self.rows[row["name"]] = row
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")


class BackboneSearch:
    def __init__(self, inp, base_spec, exp, device: JetsonDevice, search, store: ResultStore):
        self.inp, self.base, self.exp, self.device, self.cfg, self.store = inp, base_spec, exp, device, search, store
        self.metric_key = f"lat_{search.metric}"

    def measure(self, embed_dim: int, num_heads: int, depth: int) -> float:
        spec = replace(self.base, embed_dim=embed_dim, num_heads=num_heads, depth=depth)
        name = run_name(spec, self.inp, self.exp)
        if name not in self.store.rows:
            try:
                row = benchmark_spec(spec, self.inp, self.exp, self.device, self.cfg.output_dir)
            except RemoteCommandError as e:  # e.g. TRT build OOM on the device
                logger.error("[%s] failed: %s", name, str(e).splitlines()[0])
                row = {"name": name, "embed_dim": embed_dim, "num_heads": num_heads, "depth": depth, "failed": True}
            self.store.add(row)
        row = self.store.rows[name]
        lat = row.get(self.metric_key)
        status = "FAIL" if lat is None else ("fits" if lat <= self.cfg.budget_ms else "over")
        logger.info("  -> d%d L%d: %s ms (%s budget %.1f)", embed_dim, depth, lat, status, self.cfg.budget_ms)
        return math.inf if lat is None else lat

    def search_width(self, embed_dim: int, num_heads: int):
        """Deepest depth that fits at this width, or None if even min_depth is over budget."""
        fits = lambda lat: lat <= self.cfg.budget_ms  # noqa: E731
        probes = sorted(set([self.cfg.min_depth, *self.cfg.probe_depths]))
        lats = {probes[0]: self.measure(embed_dim, num_heads, probes[0])}
        if not fits(lats[probes[0]]):
            logger.info("width %d: depth %d already over budget", embed_dim, probes[0])
            return None
        for d in probes[1:]:
            lats[d] = self.measure(embed_dim, num_heads, d)
        d0, d1 = probes[0], probes[-1]
        slope = (lats[d1] - lats[d0]) / (d1 - d0) if d1 > d0 else math.nan
        if not math.isfinite(slope) or slope <= 0:
            return max(d for d, lat in lats.items() if fits(lat))
        guess = d0 + math.floor((self.cfg.budget_ms - lats[d0]) / slope)
        depth = min(max(guess, self.cfg.min_depth), self.cfg.max_depth)
        logger.info("width %d: %.2f ms/block, predicted max depth %d", embed_dim, slope, depth)

        if fits(self.measure(embed_dim, num_heads, depth)):
            while depth < self.cfg.max_depth and fits(self.measure(embed_dim, num_heads, depth + 1)):
                depth += 1
            return depth
        while depth > self.cfg.min_depth:
            depth -= 1
            if fits(self.measure(embed_dim, num_heads, depth)):
                return depth
        return self.cfg.min_depth

    def run(self):
        """Widths in increasing order; latency grows with width, so stop at the first that cannot fit."""
        best = {}
        widths = sorted(self.cfg.widths, key=lambda w: w["embed_dim"])
        for w in tqdm(widths, desc="widths", unit="width"):
            logger.info("===== width embed_dim=%d heads=%d =====", w["embed_dim"], w["num_heads"])
            best[w["embed_dim"]] = self.search_width(w["embed_dim"], w["num_heads"])
            if best[w["embed_dim"]] is None:
                logger.info("stopping: wider backbones will not fit either")
                break
        return best


def summarize(store: ResultStore, best: dict, cfg, metric_key: str) -> pd.DataFrame:
    df = pd.DataFrame([r for r in store.rows.values() if not r.get("failed")])
    df["fits_budget"] = df[metric_key] <= cfg.budget_ms
    df["frontier"] = [best.get(e) == d for e, d in zip(df.embed_dim, df.depth)]
    df = df.sort_values(["embed_dim", "depth"])
    cols = [
        "name",
        "embed_dim",
        "depth",
        "num_heads",
        "params_m",
        "tokens",
        "lat_mean",
        "lat_median",
        "lat_p95",
        "lat_p99",
        "gpu_mean",
        "fits_budget",
        "frontier",
    ]
    df[cols].to_csv(os.path.join(cfg.output_dir, "summary.csv"), index=False)
    return df[cols]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="video_classification/configs/backbone_search.yaml")
    args = p.parse_args()

    inp, base, exp, dev_cfg, search = load_search_config(args.config)
    os.makedirs(search.output_dir, exist_ok=True)
    setup_logging(os.path.join(search.output_dir, "search.log"))
    store = ResultStore(os.path.join(search.output_dir, "results.jsonl"))
    with JetsonDevice(dev_cfg) as dev:
        searcher = BackboneSearch(inp, base, exp, dev, search, store)
        best = searcher.run()

    df = summarize(store, best, search, searcher.metric_key)
    frontier = df[df.frontier].sort_values("params_m", ascending=False)
    print("\nDeepest backbone under budget per width (largest first):")
    print(frontier.to_string(index=False))
    print(f"\nFull table: {os.path.join(search.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
