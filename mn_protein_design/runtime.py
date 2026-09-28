from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_APP_HOME = PROJECT_DIR / "mn-protein-design-workdir"
DEFAULT_TMPDIR = PROJECT_DIR / ".tmp"
DEFAULT_SHARED_REFERENCE_ROOT = Path("/mnt/db/reference_files")


def ensure_runtime_home(app_home: str | Path = DEFAULT_APP_HOME, tmpdir: str | Path = DEFAULT_TMPDIR) -> Path:
    home_path = Path(app_home).expanduser().resolve()
    tmp_path = Path(tmpdir).expanduser().resolve()
    home_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (home_path / "workdir" / "runs").mkdir(parents=True, exist_ok=True)
    (home_path / "workdir" / "targets").mkdir(parents=True, exist_ok=True)
    (home_path / "reference_files").mkdir(parents=True, exist_ok=True)
    return home_path


def app_home() -> Path:
    return Path(os.getenv("MN_PROTEIN_DESIGN_APP_HOME", str(DEFAULT_APP_HOME))).expanduser().resolve()


def runs_root() -> Path:
    return Path(os.getenv("MN_PROTEIN_DESIGN_RUN_DIR", str(app_home() / "workdir" / "runs"))).expanduser().resolve()


def reference_root() -> Path:
    configured = os.getenv("MN_PROTEIN_DESIGN_REFERENCE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if DEFAULT_SHARED_REFERENCE_ROOT.exists():
        return DEFAULT_SHARED_REFERENCE_ROOT.resolve()
    return (app_home() / "reference_files").expanduser().resolve()
