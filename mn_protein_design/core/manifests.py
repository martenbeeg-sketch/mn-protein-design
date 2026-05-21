from __future__ import annotations

from pathlib import Path

import yaml


MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "manifests"


def load_manifest(tool: str) -> dict:
    path = MANIFEST_ROOT / f"{tool}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No manifest found for tool '{tool}' at {path}")
    return yaml.safe_load(path.read_text()) or {}


def list_manifests() -> list[dict]:
    manifests: list[dict] = []
    for path in sorted(MANIFEST_ROOT.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        payload["_path"] = str(path)
        manifests.append(payload)
    return manifests
