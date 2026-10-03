"""SSH session to the Jetson: upload ONNX, build a TensorRT engine, benchmark it."""

import logging
import os
import posixpath
import time
from typing import Tuple

import paramiko

from video_classification.core.config import DeviceConfig
from video_classification.deploy import trtexec

logger = logging.getLogger(__name__)

HEARTBEAT_S = 30


class RemoteCommandError(RuntimeError):
    pass


class JetsonDevice:
    def __init__(self, cfg: DeviceConfig):
        self.cfg = cfg
        self._ssh = None

    def __enter__(self):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(self.cfg.host, username=self.cfg.user, password=self.cfg.password, timeout=20)
        self.run(f"mkdir -p {self.cfg.remote_dir}")
        return self

    def __exit__(self, *exc):
        if self._ssh:
            self._ssh.close()

    def run(self, cmd: str, timeout: int = 3600, check: bool = True, step: str = None) -> Tuple[int, str]:
        """Run a remote command; when ``step`` is given, log a heartbeat while it runs."""
        chan = self._ssh.get_transport().open_session()
        chan.exec_command(cmd + " 2>&1")
        chunks, t0, next_beat = [], time.time(), HEARTBEAT_S
        while not (chan.exit_status_ready() and not chan.recv_ready()):
            if chan.recv_ready():
                chunks.append(chan.recv(65536))
                continue
            elapsed = time.time() - t0
            if elapsed > timeout:
                chan.close()
                raise RemoteCommandError(f"timeout after {timeout}s: {cmd}")
            if step and elapsed >= next_beat:
                logger.info("  ... %s still running (%ds)", step, elapsed)
                next_beat += HEARTBEAT_S
            time.sleep(0.2)
        out = b"".join(chunks).decode(errors="replace")
        code = chan.recv_exit_status()
        if step:
            logger.info("  %s done in %.0fs (exit %d)", step, time.time() - t0, code)
        if check and code != 0:
            raise RemoteCommandError(f"exit {code}: {cmd}\n{out[-3000:]}")
        return code, out

    def upload(self, local_path: str) -> str:
        remote = posixpath.join(self.cfg.remote_dir, os.path.basename(local_path))
        with self._ssh.open_sftp() as sftp:
            sftp.put(local_path, remote)
        return remote

    def benchmark_onnx(self, local_onnx: str, log_dir: str) -> trtexec.TrtResult:
        """Upload, build an engine, benchmark it; full trtexec logs are saved under log_dir."""
        stem = os.path.splitext(os.path.basename(local_onnx))[0]
        logger.info("  uploading %s (%.0f MB)", os.path.basename(local_onnx), os.path.getsize(local_onnx) / 1e6)
        remote_onnx = self.upload(local_onnx)
        remote_engine = posixpath.join(self.cfg.remote_dir, stem + ".engine")
        try:
            _, build_log = self.run(
                trtexec.build_cmd(self.cfg.trtexec, remote_onnx, remote_engine, self.cfg.build_flags), step="TRT build"
            )
            self._save(log_dir, stem + ".build.log", build_log)
            _, infer_log = self.run(
                trtexec.infer_cmd(self.cfg.trtexec, remote_engine, self.cfg.infer_flags), step="TRT benchmark"
            )
            self._save(log_dir, stem + ".infer.log", infer_log)
        except RemoteCommandError as e:
            self._save(log_dir, stem + ".error.log", str(e))
            raise
        finally:
            if not self.cfg.keep_remote_files:
                self.run(f"rm -f {remote_onnx} {remote_engine}", check=False)
        return trtexec.parse_infer_log(infer_log)

    @staticmethod
    def _save(log_dir: str, name: str, text: str):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, name), "w") as f:
            f.write(text)
