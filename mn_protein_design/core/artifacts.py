from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    type: str
    description: str = ""

    def to_json(self, run_dir: Path) -> dict:
        try:
            rel_path = self.path.relative_to(run_dir)
        except ValueError:
            rel_path = self.path
        return {"name": self.name, "path": str(rel_path), "type": self.type, "description": self.description}


def artifact_path(run_dir: Path, *parts: str) -> Path:
    path = run_dir / "artifacts" / Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_run_zip(run_dir: Path) -> bytes:
    """Create a fresh in-memory zip for a run without writing it to disk."""
    contract_names = ["input.json", "metadata.json", "command.json", "stdout.log", "stderr.log", "result.json"]
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name in contract_names:
            path = run_dir / name
            if path.exists() and path.is_file():
                archive.write(path, arcname=name)
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists():
            for path in sorted(p for p in artifacts_dir.rglob("*") if p.is_file()):
                archive.write(path, arcname=str(path.relative_to(run_dir)))
    return buffer.getvalue()
