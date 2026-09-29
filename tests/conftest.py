from __future__ import annotations

from pathlib import Path

import pytest

from mn_protein_design.core import jobs
from mn_protein_design.core import portable_paths


@pytest.fixture
def run_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point file-backed job helpers at an isolated temporary run directory."""
    root = tmp_path / "runs"
    scheduler_state = tmp_path / "scheduler-state"
    monkeypatch.setattr(jobs, "runs_root", lambda: root)
    monkeypatch.setattr(portable_paths, "runs_root", lambda: root)
    monkeypatch.setenv("MN_COMPUTE_SCHEDULER_STATE_DIR", str(scheduler_state))
    previous_locks = dict(jobs._PROCESS_RESOURCE_LOCKS)
    jobs._PROCESS_RESOURCE_LOCKS.clear()
    try:
        yield root
    finally:
        jobs._PROCESS_RESOURCE_LOCKS.clear()
        jobs._PROCESS_RESOURCE_LOCKS.update(previous_locks)
