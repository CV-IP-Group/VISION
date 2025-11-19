"""Lightweight helpers to retrieve the latest LiDAR snapshot without blocking."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import SIMULATION_MODE

# DEFAULT_SNAPSHOT_PATH = os.getenv("VISIONMATE_LIDAR_SNAPSHOT", "runs/lidar_snapshot.json")
# DEFAULT_TTL_SECONDS = float(os.getenv("VISIONMATE_LIDAR_TTL", "0.5"))
# --- NEW VERSION ---


def _current_snapshot_path() -> Path:
    """Return the latest snapshot path, honoring runtime environment overrides."""
    override = os.getenv("VISIONMATE_LIDAR_SNAPSHOT")
    return Path(override) if override else Path("runs/lidar_snapshot.json")


def _current_ttl_seconds() -> float:
    """Return the configured TTL, defaulting to 0.5s when env vars are absent or invalid."""
    raw = os.getenv("VISIONMATE_LIDAR_TTL", "0.5")
    try:
        return float(raw)
    except ValueError:
        return 0.5


def read_latest_snapshot(path: Optional[str] = None, ttl: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Return the newest LiDAR snapshot if it is fresh enough.

    The function never blocks the caller. When the snapshot file is missing,
    stale, or malformed, ``None`` is returned silently.
    """

    # snapshot_path = Path(path or DEFAULT_SNAPSHOT_PATH)
    # expiry_seconds = DEFAULT_TTL_SECONDS if ttl is None else ttl
    # --- NEW VERSION ---
    snapshot_path = Path(path) if path else _current_snapshot_path()
    expiry_seconds = _current_ttl_seconds() if ttl is None else ttl

    try:
        stat = snapshot_path.stat()
    except FileNotFoundError:
        if not SIMULATION_MODE:
            logging.warning("Real mode: LiDAR snapshot missing at %s", snapshot_path)
        return None
    except OSError:
        if not SIMULATION_MODE:
            logging.warning("Real mode: Unable to stat LiDAR snapshot at %s", snapshot_path)
        return None

    age = time.time() - stat.st_mtime
    if age > expiry_seconds:
        if not SIMULATION_MODE:
            logging.warning(
                "Real mode: LiDAR snapshot at %s is too old (%.2fs > %.2fs)",
                snapshot_path,
                age,
                expiry_seconds,
            )
        return None

    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        if not SIMULATION_MODE:
            logging.warning("Real mode: LiDAR snapshot at %s is unreadable", snapshot_path)
        return None

    if not SIMULATION_MODE and isinstance(payload, dict) and payload.get("source") == "mock":
        logging.warning("Real mode: Ignoring mock LiDAR payload at %s", snapshot_path)
        return None

    return payload
