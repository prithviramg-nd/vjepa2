"""
Video download via AVC API + AWS S3.
"""

import os
import subprocess
from typing import List, Optional

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import AVC_API_ENDPOINT


def _retry_session(retries=5, backoff_factor=1,
                   status_forcelist=(400, 429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries,
                  backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def query_avc_api(avid: str, env: str = "production") -> Optional[dict]:
    """Query AVC API to get S3 paths for an AVID."""
    api_env = "secondary" if env == "staging" else "primary"
    params = {
        "input_data": {"avid": avid},
        "anonymize_environment": api_env,
        "source": "debug",
        "api_version": "v2",
    }
    logger.info(f"[{avid}] Querying AVC API...")
    try:
        resp = _retry_session().post(AVC_API_ENDPOINT, json=params, timeout=60)
        result = resp.json()
        if result.get("msg") == "success":
            logger.info(f"[{avid}] AVC API success")
            return result
        else:
            logger.warning(f"[{avid}] AVC API response: {result.get('msg', result)}")
            return None
    except Exception as e:
        logger.exception(f"[{avid}] AVC API error: {e}")
        return None


def _s3_exists(s3_path: str) -> bool:
    try:
        r = subprocess.run(["aws", "s3", "ls", s3_path],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False


def _s3_cp(s3_path: str, local_path: str) -> bool:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        r = subprocess.run(["aws", "s3", "cp", s3_path, local_path],
                           capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


def _s3_ls(s3_path: str) -> List[str]:
    try:
        r = subprocess.run(["aws", "s3", "ls", s3_path, "--recursive"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return []
        keys = []
        for line in r.stdout.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 4:
                keys.append(parts[-1])
        return keys
    except Exception:
        return []


def download_dms_video(avid: str, api_result: dict, download_dir: str) -> Optional[str]:
    """
    Download DMS video (8.mp4 or dmsVideo.mp4) for an AVID.
    Returns local path on success, None on failure.
    """
    s3_path_list = api_result.get("s3_bucket")
    if s3_path_list is None:
        logger.error(f"[{avid}] No s3_bucket in API response")
        return None

    if not isinstance(s3_path_list, list):
        s3_path_list = [s3_path_list]

    for s3_path in s3_path_list:
        tokens = s3_path.split("/")
        bucket = tokens[0]
        prefix = "/".join(tokens[1:]).rstrip("/")
        s3_base = f"s3://{bucket}/{prefix}"

        # Try direct paths first
        for filename in ("8.mp4", "dmsVideo.mp4"):
            s3_direct = f"{s3_base}/{filename}"
            logger.debug(f"[{avid}] Trying: {s3_direct}")
            if _s3_exists(s3_direct):
                local_path = os.path.join(download_dir, avid, "dmsVideo.mp4")
                if _s3_cp(s3_direct, local_path):
                    logger.info(f"[{avid}] Downloaded -> {local_path}")
                    return local_path

        # Fallback: list and search
        logger.debug(f"[{avid}] Fallback: listing {s3_base}/")
        keys = _s3_ls(s3_base + "/")
        for key in keys:
            if key.split("/")[-1] in ("dmsVideo.mp4", "8.mp4"):
                s3_full = f"s3://{bucket}/{key}"
                local_path = os.path.join(download_dir, avid, "dmsVideo.mp4")
                if _s3_cp(s3_full, local_path):
                    logger.info(f"[{avid}] Downloaded -> {local_path}")
                    return local_path

    logger.error(f"[{avid}] Could not find/download DMS video")
    return None
