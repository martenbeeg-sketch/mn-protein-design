from __future__ import annotations

import subprocess
from pathlib import Path

from mn_protein_design.core.jobs import update_status, write_json


def build_docker_command(manifest: dict, run_dir: Path, args: list[str], reference_root: Path = Path("/mnt/db/reference_files")) -> list[str]:
    command = ["docker", "run", "--rm"]
    if manifest.get("gpu"):
        command.extend(["--gpus", "all"])
    command.extend(["-v", f"{run_dir}:/work"])
    if reference_root.exists():
        command.extend(["-v", f"{reference_root}:/ref:ro"])
    command.append(str(manifest["image"]))
    command.extend(args)
    return command


def run_docker_job(run_dir: Path, manifest: dict, args: list[str]) -> int:
    command = build_docker_command(manifest, run_dir, args)
    write_json(run_dir / "command.json", {"mode": "docker", "command": command, "image": manifest.get("image")})
    update_status(run_dir, "running")
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    return int(proc.returncode)
