"""Run-manifest utilities for reproducibility."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_run_manifest(path: str | Path, *, seed: int, extra: dict[str, Any] | None = None) -> None:
    """Write environment and command-line metadata as JSON."""

    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "seed": seed,
        "packages": {
            "numpy": _package_version("numpy"),
            "pytest": _package_version("pytest"),
            "walsh-haar-fcgr": _package_version("walsh-haar-fcgr"),
        },
    }
    if extra:
        payload["extra"] = extra
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
