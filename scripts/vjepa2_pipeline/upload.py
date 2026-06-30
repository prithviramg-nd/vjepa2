"""
S3 upload for embedding pickle files.
"""

import os
import subprocess

from loguru import logger


def upload_to_s3(local_path: str, s3_path: str) -> bool:
    """Upload a file to S3. Returns True on success."""
    try:
        logger.info(f"Uploading {local_path} -> {s3_path}")
        r = subprocess.run(
            ["aws", "s3", "cp", local_path, s3_path],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0:
            logger.info(f"Uploaded -> {s3_path}")
            return True
        else:
            logger.error(f"Upload failed: {r.stderr.strip()}")
            return False
    except Exception as e:
        logger.exception(f"Upload error: {e}")
        return False


def build_s3_key(s3_base: str, avid: str, fps: int) -> str:
    """
    Build S3 path for an embedding file.
    Structure: s3_base/<avid>/<fps>fps_embedding.pkl
    """
    return f"{s3_base.rstrip('/')}/{avid}/{fps}fps_embedding.pkl"
