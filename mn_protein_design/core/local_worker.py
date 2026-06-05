from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from mn_protein_design.core.jobs import JobPaths, finish_job, read_json, update_status


REPO_ROOT = Path(__file__).resolve().parents[2]


def _worker_log(run_dir: Path, name: str, text: str) -> None:
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _restore_worker_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(payload.get("kwargs") or {})
    path_kwargs = set(payload.get("path_kwargs") or [])
    for key in path_kwargs:
        value = kwargs.get(key)
        if value not in {None, ""}:
            kwargs[key] = Path(str(value)).expanduser()
    kwargs.pop("progress_callback", None)
    kwargs.pop("existing_job", None)
    return kwargs


def run_worker_job(run_dir: Path) -> int:
    run_dir = run_dir.expanduser().resolve()
    request = read_json(run_dir / "worker_request.json")
    kind = str(request.get("kind") or "").strip()
    if not kind:
        raise RuntimeError(f"No worker_request.json found for {run_dir}")

    try:
        if kind in {"de_novo_binder_scoring_dataset", "candidate_refolding_evaluation"}:
            from mn_protein_design.workflows.benchmark import run_de_novo_binder_scoring_dataset

            kwargs = _restore_worker_kwargs(request)
            job = JobPaths(task_group="benchmark", run_id=run_dir.name, run_dir=run_dir)
            run_de_novo_binder_scoring_dataset(existing_job=job, **kwargs)
            return 0
        raise RuntimeError(f"Unknown local worker request kind: {kind}")
    except Exception as exc:
        _worker_log(run_dir, "worker_stderr.log", traceback.format_exc())
        result = read_json(run_dir / "result.json")
        if not result:
            finish_job(
                run_dir,
                False,
                {
                    "metrics": {
                        "worker_error": str(exc),
                        "worker_exception_type": type(exc).__name__,
                    }
                },
            )
        else:
            update_status(run_dir, "failed", worker_error=str(exc), worker_exception_type=type(exc).__name__)
        return 1


def spawn_worker_for_run(run_dir: Path) -> subprocess.Popen:
    run_dir = run_dir.expanduser().resolve()
    stdout = (run_dir / "worker_stdout.log").open("a")
    stderr = (run_dir / "worker_stderr.log").open("a")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mn_protein_design.core.local_worker",
            "--run-dir",
            str(run_dir),
        ],
        cwd=str(REPO_ROOT),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a queued mn-protein-design local worker job.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run_worker_job(args.run_dir))


if __name__ == "__main__":
    main()
