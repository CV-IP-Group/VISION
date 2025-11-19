from __future__ import annotations

import os


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


SIMULATION_MODE: bool = _parse_bool(os.getenv("VISIONMATE_SIMULATION_MODE", "True"))
