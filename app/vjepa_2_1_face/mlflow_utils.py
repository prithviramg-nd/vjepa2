"""Thin MLflow wrapper for the V-JEPA 2.1 face pipeline.

Designed so that:
  * only rank 0 ever talks to MLflow;
  * every call is a no-op (never raises) if MLflow is disabled or unavailable,
    so a tracking problem can never kill a long training run.
"""

import logging
import os

logger = logging.getLogger()


class MLflowRun:
    def __init__(self, cfg, rank=0, enabled=True):
        self.active = False
        self._mlflow = None
        if not enabled or rank != 0 or cfg is None or not cfg.get("enabled", True):
            return
        try:
            import mlflow

            uri = cfg.get("tracking_uri")
            if uri:
                mlflow.set_tracking_uri(uri)

            # MLflow >=3 refuses a bare file store, so the config uses sqlite. Artifacts
            # still need an explicit home, set once at experiment-creation time.
            exp_name = cfg.get("experiment", "vjepa2_1_face")
            artifact_location = cfg.get("artifact_location")
            if mlflow.get_experiment_by_name(exp_name) is None and artifact_location:
                mlflow.create_experiment(exp_name, artifact_location=artifact_location)
            mlflow.set_experiment(exp_name)

            run_name = cfg.get("run_name") or None
            mlflow.start_run(run_name=run_name)
            self._mlflow = mlflow
            self.active = True
            logger.info(f"[mlflow] logging to {mlflow.get_tracking_uri()} exp={exp_name}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[mlflow] disabled ({type(e).__name__}: {e})")

    def _flatten(self, d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(self._flatten(v, prefix=f"{key}."))
            else:
                out[key] = str(v)[:490]
        return out

    def log_params(self, params: dict):
        if not self.active:
            return
        try:
            flat = self._flatten(params)
            items = list(flat.items())
            for i in range(0, len(items), 100):  # MLflow caps batch size
                self._mlflow.log_params(dict(items[i : i + 100]))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[mlflow] log_params failed: {e}")

    def log_metrics(self, metrics: dict, step: int):
        if not self.active:
            return
        try:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(clean, step=int(step))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[mlflow] log_metrics failed: {e}")

    def log_artifact(self, path: str, artifact_path=None):
        if not self.active or not os.path.exists(path):
            return
        try:
            self._mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[mlflow] log_artifact failed: {e}")

    def set_tag(self, key, value):
        if not self.active:
            return
        try:
            self._mlflow.set_tag(key, value)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[mlflow] set_tag failed: {e}")

    def close(self, status="FINISHED"):
        if not self.active:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception:  # noqa: BLE001
            pass
        self.active = False
