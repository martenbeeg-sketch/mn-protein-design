from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mn_protein_design.core.jobs import utc_now, write_json
from mn_protein_design.runtime import runs_root


PRESET_SCHEMA_VERSION = "mn-protein-design.benchmark-feature-preset.v1"


def preset_root() -> Path:
    root = runs_root() / "_presets" / "benchmark_feature_presets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_preset_name(name: str) -> str:
    text = str(name or "benchmark_feature_preset").strip() or "benchmark_feature_preset"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    safe = safe.strip("._-")
    return safe or "benchmark_feature_preset"


def preset_path(name: str) -> Path:
    return preset_root() / f"{safe_preset_name(name)}.json"


def save_feature_preset(
    *,
    name: str,
    source_run_dir: Path,
    rows: list[dict[str, Any]],
    description: str = "",
    selection_rule: str = "best_average_precision_per_engine",
) -> Path:
    source_run_dir = Path(source_run_dir)
    payload = {
        "schema_version": PRESET_SCHEMA_VERSION,
        "name": name,
        "description": description,
        "selection_rule": selection_rule,
        "source_task_group": source_run_dir.parent.name,
        "source_run_id": source_run_dir.name,
        "source_run_dir": str(source_run_dir),
        "created_at": utc_now(),
        "features": rows,
    }
    path = preset_path(name)
    write_json(path, payload)
    return path


def load_feature_presets() -> list[dict[str, Any]]:
    presets: list[dict[str, Any]] = []
    for path in sorted(preset_root().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("schema_version") != PRESET_SCHEMA_VERSION:
            continue
        payload["_path"] = str(path)
        presets.append(payload)
    return presets


def load_feature_preset(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    if payload.get("schema_version") != PRESET_SCHEMA_VERSION:
        return {}
    payload["_path"] = str(path)
    return payload
