"""Provenance helpers: software versions, device info, git state, hashing.

Everything needed to tie a saved model / prediction file back to the exact
code, data and parameters that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


def _optional_version(module_name: str) -> str | None:
    import importlib.metadata as importlib_metadata

    try:
        return importlib_metadata.version(module_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": None
        if run(["rev-parse", "HEAD"]) is None
        else bool(run(["status", "--porcelain"])),
        "remote": run(["config", "--get", "remote.origin.url"]),
    }


def collect_environment() -> dict[str, Any]:
    """Collect versions of the core scientific stack and accelerator info."""
    import torch

    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "macos": platform.mac_ver()[0] if platform.system() == "Darwin" else None,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None)
        if torch.cuda.is_available()
        else None,
        "gpus": [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_mb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1e6, 1
                ),
            }
            for i in range(torch.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        # Reported for completeness; MPS is NOT used automatically (see config).
        "mps_available": bool(getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()),
        "numpy": _optional_version("numpy"),
        "scipy": _optional_version("scipy"),
        "pandas": _optional_version("pandas"),
        "sklearn": _optional_version("scikit-learn"),
        "anndata": _optional_version("anndata"),
        "scanpy": _optional_version("scanpy"),
        "scvi": _optional_version("scvi-tools"),
        "lightning": _optional_version("lightning")
        or _optional_version("pytorch-lightning"),
    }
    env["git"] = _git_info()
    return env


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash_strings(values) -> str:
    """Order-independent SHA256 over a collection of strings."""
    h = hashlib.sha256()
    for v in sorted(str(x) for x in values):
        h.update(v.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def dataset_fingerprint(adata) -> dict[str, Any]:
    """Cheap fingerprint of the loaded AnnData (no full-matrix hashing)."""
    n_obs, n_vars = adata.shape
    var_hash = stable_hash_strings(list(adata.var_names))
    obs_hash = stable_hash_strings(list(adata.obs_names))
    return {
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
        "obs_names_sha256": obs_hash,
        "var_names_sha256": var_hash,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)
