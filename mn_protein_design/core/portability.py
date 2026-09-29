from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from mn_protein_design.core.portable_paths import (
    PORTABLE_SCHEMES,
    is_operational_path_key,
    is_provenance_key,
    resolve_stored_path,
)
from mn_protein_design.runtime import app_home, reference_root, runs_root


EXPORT_MANIFEST = "portable-export.json"
EXPORT_REPORT = "migration-report.json"
EXPORT_INSTRUCTIONS = "IMPORT.md"
EXPORT_SCHEMA_VERSION = 1
PORTABLE_PREFIXES = tuple(f"{scheme}:///" for scheme in sorted(PORTABLE_SCHEMES))
CONTAINER_PREFIXES = (
    "/app/", "/cache/", "/data/", "/input/", "/models/", "/opt/",
    "/output/", "/reference/", "/references/", "/source/", "/work/", "/workspace/",
)
PROVENANCE_FILENAMES = frozenset({"command.json"})


class PortabilityError(RuntimeError):
    """Raised when a workdir cannot be safely exported or imported."""


@dataclass(frozen=True)
class _Roots:
    app_home: Path
    workdir: Path
    runs: Path
    references: Path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative(value: str) -> Path | None:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    return Path(*relative.parts)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _roots(
    *,
    app_home_path: Path | None = None,
    runs_path: Path | None = None,
    references_path: Path | None = None,
) -> _Roots:
    home = Path(app_home_path or app_home()).expanduser().resolve()
    workdir = (home / "workdir").resolve()
    runs = Path(runs_path or runs_root()).expanduser().resolve()
    references = Path(references_path or reference_root()).expanduser().resolve()
    if runs == workdir:
        raise PortabilityError("The runs directory cannot be the entire workdir.")
    if _is_relative_to(workdir, runs):
        raise PortabilityError("The workdir cannot be inside the runs directory.")
    if _is_relative_to(runs, references) or _is_relative_to(references, runs):
        raise PortabilityError("The runs and reference roots cannot contain one another.")
    if references == workdir:
        raise PortabilityError("The reference root cannot be the entire workdir.")
    return _Roots(home, workdir, runs, references)


def _iter_data_files(root: Path, *, exclude_roots: tuple[Path, ...] = ()) -> list[Path]:
    files: list[Path] = []
    excluded = tuple(path.resolve() for path in exclude_roots)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        kept_dirs: list[str] = []
        for name in dirnames:
            path = current / name
            resolved = path.resolve()
            if any(resolved == item or _is_relative_to(resolved, item) for item in excluded):
                continue
            if path.is_symlink():
                raise PortabilityError(f"Portable workdir contains a symbolic link: {path}")
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in filenames:
            path = current / name
            resolved = path.resolve()
            if any(resolved == item or _is_relative_to(resolved, item) for item in excluded):
                continue
            if path.is_symlink():
                raise PortabilityError(f"Portable workdir contains a symbolic link: {path}")
            if path.is_file():
                files.append(path)
    return sorted(files)


def _metadata_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            if filename.endswith((".json", ".jsonl")):
                paths.append(Path(directory) / filename)
    return sorted(paths)


def _all_source_metadata(roots: _Roots) -> list[Path]:
    excluded = (roots.references,) if _is_relative_to(roots.references, roots.workdir) else ()
    files = set()
    if roots.workdir.is_dir():
        files = {path for path in _metadata_files(roots.workdir) if not any(_is_relative_to(path.resolve(), root) for root in excluded)}
    if roots.runs.is_dir():
        files.update(_metadata_files(roots.runs))
    return sorted(files)


def _active_runs(runs: Path) -> list[tuple[Path, str]]:
    active: list[tuple[Path, str]] = []
    if not runs.is_dir():
        return active
    for metadata_path in runs.glob("*/*/metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text())
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status in {"queued", "preparing", "running", "paused", "holding"}:
            active.append((metadata_path.parent, status))
    return active


def _configured_and_legacy_run_roots(roots: _Roots) -> list[Path]:
    candidates = [roots.runs, roots.workdir / "runs"]
    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _active_jobs(roots: _Roots) -> list[tuple[Path, str]]:
    active: list[tuple[Path, str]] = []
    for run_root in _configured_and_legacy_run_roots(roots):
        active.extend(_active_runs(run_root))
    return active


def _portable_uri(raw: str) -> tuple[str, str] | None:
    parsed = urlsplit(raw)
    if parsed.scheme not in PORTABLE_SCHEMES:
        return None
    joined = "/".join(part for part in (parsed.netloc, parsed.path) if part)
    relative = _safe_relative(joined.lstrip("/"))
    if relative is None:
        return None
    return parsed.scheme, relative.as_posix()


def _map_source_path(raw: str, roots: _Roots) -> tuple[str, Path | None] | None:
    existing_uri = _portable_uri(raw)
    if existing_uri is not None:
        scheme, relative = existing_uri
        if scheme == "reference":
            return f"reference:///{relative}", None
        base = roots.runs if scheme == "runs" else roots.app_home
        candidate = (base / relative).resolve()
        if scheme == "app" and not _is_relative_to(candidate, roots.workdir):
            return None
        if not candidate.exists():
            return None
        return f"{scheme}:///{relative}", candidate
    if urlsplit(raw).scheme:
        return None
    source = Path(raw).expanduser()
    if source.is_absolute():
        normalized = source.resolve()
    else:
        return None

    # References are intentionally not copied. Their URI keeps the reference
    # to data that the destination can receive separately.
    try:
        relative = normalized.relative_to(roots.references).as_posix()
    except ValueError:
        pass
    else:
        if relative and relative != ".":
            return f"reference:///{relative}", None

    try:
        relative = normalized.relative_to(roots.runs).as_posix()
    except ValueError:
        pass
    else:
        if relative and relative != "." and normalized.exists():
            return f"runs:///{relative}", normalized

    try:
        relative = normalized.relative_to(roots.workdir).as_posix()
    except ValueError:
        pass
    else:
        if relative and normalized.exists():
            return f"app:///workdir/{relative}", normalized

    # Resolve known old absolute paths if they were moved under the active run
    # or workdir roots before this export.
    relocated = resolve_stored_path(raw, must_exist=True)
    if relocated is not None:
        candidate = relocated.resolve()
        try:
            relative = candidate.relative_to(roots.runs).as_posix()
        except ValueError:
            pass
        else:
            return f"runs:///{relative}", candidate
        try:
            relative = candidate.relative_to(roots.workdir).as_posix()
        except ValueError:
            pass
        else:
            return f"app:///workdir/{relative}", candidate
    return None


def _operational_key(key: str) -> bool:
    return is_operational_path_key(key)


def _provenance_key(key: str) -> bool:
    return is_provenance_key(key)


def _strings(value: Any, key: str = ""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _strings(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child, key)
    elif isinstance(value, str):
        yield key, value


def _rewrite_value(value: Any, *, roots: _Roots, relative_file: str, report: dict[str, Any], key: str = "", provenance: bool = False) -> Any:
    provenance = provenance or _provenance_key(key)
    if isinstance(value, dict):
        return {
            child_key: _rewrite_value(child, roots=roots, relative_file=relative_file, report=report, key=str(child_key), provenance=provenance)
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_value(child, roots=roots, relative_file=relative_file, report=report, key=key, provenance=provenance) for child in value]
    if not isinstance(value, str) or provenance or not _operational_key(key):
        return value
    if not value.startswith("/"):
        return value
    mapped = _map_source_path(value, roots)
    if mapped is None:
        if value.startswith(CONTAINER_PREFIXES):
            return value
        report["unresolved"].append({"file": relative_file, "key": key})
        return value
    portable, _ = mapped
    report["rewritten"].append({"file": relative_file, "key": key, "portable": portable})
    return portable


def _rewrite_metadata_file(path: Path, *, bundle: Path, roots: _Roots, report: dict[str, Any]) -> None:
    if path.name in PROVENANCE_FILENAMES:
        return
    relative = path.relative_to(bundle).as_posix()
    raw_text = path.read_text()
    if path.suffix == ".jsonl":
        rows: list[str] = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except (ValueError, TypeError) as exc:
                report["unresolved"].append({"file": relative, "key": f"line:{line_number}", "reason": f"invalid JSONL: {exc}"})
                continue
            rows.append(json.dumps(_rewrite_value(payload, roots=roots, relative_file=relative, report=report), sort_keys=True))
        path.write_text("".join(row + "\n" for row in rows))
        return
    try:
        payload = json.loads(raw_text)
    except (ValueError, TypeError) as exc:
        report["unresolved"].append({"file": relative, "key": "", "reason": f"invalid JSON: {exc}"})
        return
    if isinstance(payload, (dict, list)):
        _atomic_json(path, _rewrite_value(payload, roots=roots, relative_file=relative, report=report))


def _ignore_copy(source_root: Path, excluded_roots: tuple[Path, ...], top_level_names: set[str] | None = None):
    excluded = tuple(path.resolve() for path in excluded_roots)

    def ignore(directory: str, names: list[str]) -> set[str]:
        base = Path(directory).resolve()
        ignored: set[str] = set()
        for name in names:
            path = base / name
            if base == source_root and top_level_names and name in top_level_names:
                ignored.add(name)
                continue
            resolved = path.resolve()
            if any(resolved == root or _is_relative_to(resolved, root) for root in excluded):
                ignored.add(name)
        return ignored

    return ignore


def _copy_tree(source: Path, destination: Path, *, excluded_roots: tuple[Path, ...] = (), omit_service_state: bool = False) -> None:
    top_level = {"_worker_service", "_locks"} if omit_service_state else None
    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=_ignore_copy(source, excluded_roots, top_level))


def _remove_worker_state(runs: Path) -> None:
    for name in ("_worker_service", "_locks"):
        path = runs / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def _merge_run_tree(source: Path, destination: Path) -> None:
    """Merge run files without silently replacing a different historical file."""
    destination.mkdir(parents=True, exist_ok=True)
    _remove_worker_state(destination)
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(directory)
        relative_dir = current.relative_to(source)
        target_dir = destination / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        if relative_dir == Path("."):
            dirnames[:] = [name for name in dirnames if name not in {"_worker_service", "_locks"}]
        for name in filenames:
            source_file = current / name
            relative_file = source_file.relative_to(source)
            target_file = destination / relative_file
            if target_file.exists():
                if not target_file.is_file() or _sha256(target_file) != _sha256(source_file):
                    raise PortabilityError(f"Configured and workdir runs contain conflicting files: {relative_file}")
                continue
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)


def _copy_workdir(roots: _Roots, destination: Path) -> str:
    if not roots.workdir.is_dir():
        raise PortabilityError(f"Work directory does not exist: {roots.workdir}")
    if not roots.runs.is_dir():
        raise PortabilityError(f"Runs directory does not exist: {roots.runs}")
    if not roots.references.is_dir():
        raise PortabilityError(f"Reference root is not configured or does not exist: {roots.references}")
    if roots.runs == roots.workdir:
        raise PortabilityError("Runs directory cannot be the entire workdir.")
    nested_refs = _is_relative_to(roots.references, roots.workdir)
    # Exclude the configured roots even when they are mounted into the
    # workdir through symlinks; their contents are copied/handled separately.
    excluded = (roots.runs, roots.references)
    _iter_data_files(roots.workdir, exclude_roots=excluded)
    _iter_data_files(roots.runs)
    _copy_tree(roots.workdir, destination / "workdir", excluded_roots=excluded)

    # Install all runs in the app's canonical location so the Jobs page reads
    # the complete history after import. Existing default runs are merged with
    # the configured run root, with byte-level conflict checks.
    standard_runs = destination / "workdir" / "runs"
    runs_relative = "workdir/runs"
    runs_destination = standard_runs
    if runs_destination.exists():
        _remove_worker_state(runs_destination)
        _merge_run_tree(roots.runs, runs_destination)
    else:
        _copy_tree(roots.runs, runs_destination, omit_service_state=True)
    if nested_refs:
        # The reference root is excluded from the workdir copy and restored as
        # an empty mount point; its files are intentionally a separate transfer.
        reference_relative = roots.references.relative_to(roots.workdir)
        (destination / "workdir" / reference_relative).mkdir(parents=True, exist_ok=True)
    return runs_relative


def _inventory(bundle: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in _iter_data_files(bundle):
        relative = path.relative_to(bundle).as_posix()
        if relative == EXPORT_MANIFEST:
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    return files


def audit_runtime_portability(
    *, app_home_path: Path | None = None, runs_path: Path | None = None, references_path: Path | None = None,
) -> dict[str, Any]:
    roots = _roots(app_home_path=app_home_path, runs_path=runs_path, references_path=references_path)
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {"in_workdir": [], "reference_root": [], "external": []}
    sources = _all_source_metadata(roots)
    for path in sources:
        try:
            values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(path.read_text())]
        except (OSError, ValueError, TypeError):
            counts["unreadable_metadata_files"] += 1
            continue
        if path.name in PROVENANCE_FILENAMES:
            continue
        for value in values:
            for key, raw in _strings(value):
                if not _operational_key(key) or not raw.startswith("/") or _provenance_key(key):
                    continue
                mapped = _map_source_path(raw, roots)
                if mapped is None and raw.startswith(CONTAINER_PREFIXES):
                    continue
                counts["absolute_operational_paths"] += 1
                if mapped:
                    scheme = mapped[0].split(":", 1)[0]
                    bucket = "reference_root" if scheme == "reference" else "in_workdir"
                else:
                    bucket = "external"
                counts[f"{bucket}_paths"] += 1
                if len(examples[bucket]) < 20:
                    base = roots.workdir if _is_relative_to(path, roots.workdir) else roots.runs
                    examples[bucket].append({"file": path.relative_to(base).as_posix(), "key": key, "path": raw})
    active = _active_jobs(roots)
    workdir_bytes = sum(path.stat().st_size for path in _iter_data_files(roots.workdir, exclude_roots=(roots.runs, roots.references))) if roots.workdir.is_dir() else 0
    runs_bytes = sum(path.stat().st_size for path in _iter_data_files(roots.runs)) if roots.runs.is_dir() else 0
    if _is_relative_to(roots.runs, roots.workdir):
        workdir_bytes += runs_bytes
        runs_bytes = 0
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "read_only": True,
        "app_home": str(roots.app_home),
        "workdir": str(roots.workdir),
        "runs_dir": str(roots.runs),
        "reference_dir": str(roots.references),
        "counts": dict(sorted(counts.items())),
        "examples": examples,
        "estimated_bytes": {"workdir_and_runs": workdir_bytes + runs_bytes, "reference_root_excluded": True},
        "active_jobs": [{"task_group": path.parent.name, "run_id": path.name, "status": status} for path, status in active],
    }


def export_portable_workdir(
    destination: Path,
    *, app_home_path: Path | None = None, runs_path: Path | None = None, references_path: Path | None = None,
) -> dict[str, Any]:
    """Export all workdir data and configured runs; reference files stay external."""
    roots = _roots(app_home_path=app_home_path, runs_path=runs_path, references_path=references_path)
    requested_target = Path(destination).expanduser()
    if requested_target.exists() or requested_target.is_symlink():
        raise PortabilityError(f"Export destination already exists: {requested_target}")
    target = requested_target.resolve()
    if target.exists():
        raise PortabilityError(f"Export destination already exists: {target}")
    if not roots.workdir.is_dir():
        raise PortabilityError(f"Work directory does not exist: {roots.workdir}")
    if not roots.runs.is_dir():
        raise PortabilityError(f"Runs directory does not exist: {roots.runs}")
    if not roots.references.is_dir():
        raise PortabilityError(f"Reference root is missing: {roots.references}")
    if any(_is_relative_to(target, source) for source in (roots.workdir, roots.runs, roots.references)):
        raise PortabilityError("Export destination cannot be inside a source data directory.")
    active = _active_jobs(roots)
    if active:
        examples = ", ".join(f"{path.parent.name}/{path.name} ({status})" for path, status in active[:5])
        raise PortabilityError(f"Export refused while {len(active)} job(s) are queued, active, or paused: {examples}. Let them finish first.")

    # Fail early on symlinks, except a reference root that is deliberately
    # nested in workdir and excluded from this bundle.
    excluded_workdir_roots = (roots.runs, roots.references)
    _iter_data_files(roots.workdir, exclude_roots=excluded_workdir_roots)
    _iter_data_files(roots.runs)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir()
    metadata_hashes = {path: _sha256(path) for path in _all_source_metadata(roots)}
    report: dict[str, Any] = {"rewritten": [], "unresolved": []}
    try:
        _copy_workdir(roots, staging)
        # Rewrite runtime JSON/JSONL within the workdir copy, including the
        # merged canonical runs tree.
        for path in _metadata_files(staging / "workdir"):
            _rewrite_metadata_file(path, bundle=staging, roots=roots, report=report)
        active_after = _active_jobs(roots)
        if active_after:
            raise PortabilityError("A job became active during export; stop it and retry.")
        changed = [path for path, digest in metadata_hashes.items() if not path.is_file() or _sha256(path) != digest]
        if changed:
            raise PortabilityError("Source metadata changed during export. Stop app activity and retry.")
        if report["unresolved"]:
            first = report["unresolved"][0]
            raise PortabilityError(
                f"Export found {len(report['unresolved'])} absolute operational path(s) outside the workdir/runs/reference roots; "
                f"first is {first['file']} ({first['key']}). Run portability audit for details."
            )
        layout = {"workdir": "workdir", "runs": "workdir/runs", "references": "external"}
        _atomic_json(staging / EXPORT_REPORT, {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "source_metadata_unchanged": True,
            "rewritten_path_count": len(report["rewritten"]),
            "rewritten_by_file": dict(Counter(item["file"] for item in report["rewritten"])),
            "reference_files_included": False,
            "unresolved_path_count": 0,
        })
        instructions = """# Import mn-protein-design workdir data

This bundle contains the complete workdir tree and configured run results.
Reference files/model weights and Docker images are intentionally excluded.

On the destination, install the app and its Python/Docker dependencies
separately, then run:

```bash
mn-protein-design portability verify /path/to/this/bundle
mn-protein-design portability import /path/to/this/bundle --app-home /path/to/new/app-home
```

The import prints the environment paths for the new install. Configure the
reference directory separately after copying/mounting it:

```bash
mn-protein-design app --app-home /path/to/new/app-home \\
  --runs-dir /path/to/new/app-home/workdir/runs \\
  --reference-dir /path/to/reference_files
```

The source workdir is not modified. Existing destination paths are never
overwritten. Review `migration-report.json` after export and before transfer.
"""
        (staging / EXPORT_INSTRUCTIONS).write_text(instructions)
        manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "layout": layout,
            "external_roots": {"reference": "Copy or mount reference_files separately and set MN_PROTEIN_DESIGN_REFERENCE_DIR."},
            "counts": {"rewritten_paths": len(report["rewritten"]), "unresolved_paths": 0},
            "files": _inventory(staging),
        }
        _atomic_json(staging / EXPORT_MANIFEST, manifest)
        verification = verify_portable_export(staging)
        if not verification["valid"]:
            raise PortabilityError(f"Staged workdir failed verification with {len(verification['errors'])} error(s).")
        staging.rename(target)
        return {"destination": str(target), "counts": manifest["counts"], "verification": verification}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _bundle_path(bundle: Path, raw: str) -> Path | None:
    relative = _safe_relative(raw)
    if relative is None:
        return None
    base = bundle.resolve()
    candidate = (base / relative).resolve()
    return candidate if _is_relative_to(candidate, base) else None


def _resolve_bundle_uri(raw: str, bundle: Path, layout: dict[str, Any]) -> tuple[Path | None, bool]:
    parsed = _portable_uri(raw)
    if parsed is None:
        return None, False
    scheme, relative = parsed
    if scheme == "reference":
        return None, True
    base_key = {"runs": "runs", "app": "workdir"}[scheme]
    base = _bundle_path(bundle, str(layout.get(base_key) or ""))
    suffix = _safe_relative(relative)
    if base is None or suffix is None:
        return None, False
    if scheme == "app":
        # app:/// paths are rooted at app_home and must remain inside workdir.
        workdir = _bundle_path(bundle, str(layout.get("workdir") or ""))
        candidate = (bundle / suffix).resolve()
        if workdir is None or not _is_relative_to(candidate, workdir):
            return None, False
        base = bundle
    result = (base / suffix).resolve()
    return (result if _is_relative_to(result, base) else None), False


def verify_portable_export(destination: Path) -> dict[str, Any]:
    bundle = Path(destination).expanduser().resolve()
    errors: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    try:
        manifest = json.loads((bundle / EXPORT_MANIFEST).read_text())
    except (OSError, TypeError, ValueError) as exc:
        return {"destination": str(bundle), "valid": False, "counts": {}, "errors": [{"file": EXPORT_MANIFEST, "error": str(exc)}]}
    if not isinstance(manifest, dict):
        return {"destination": str(bundle), "valid": False, "counts": {}, "errors": [{"file": EXPORT_MANIFEST, "error": "manifest must be a JSON object"}]}
    try:
        schema_version = int(manifest.get("schema_version") or 0)
    except (TypeError, ValueError):
        schema_version = 0
    if schema_version != EXPORT_SCHEMA_VERSION:
        errors.append({"file": EXPORT_MANIFEST, "error": "unsupported schema version"})
    layout = manifest.get("layout") if isinstance(manifest.get("layout"), dict) else {}
    listed: set[str] = set()
    file_entries = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    for entry in file_entries:
        if not isinstance(entry, dict):
            errors.append({"file": EXPORT_MANIFEST, "error": "invalid checksum record"})
            continue
        raw = str(entry.get("path") or "")
        path = _bundle_path(bundle, raw)
        if raw in listed:
            errors.append({"file": raw, "error": "duplicate checksum record"})
            continue
        listed.add(raw)
        if path is None or not path.is_file() or path.is_symlink():
            errors.append({"file": raw, "error": "missing or unsafe file"})
            continue
        counts["files"] += 1
        try:
            expected_size = int(entry.get("size", -1))
        except (TypeError, ValueError):
            expected_size = -1
        if path.stat().st_size != expected_size:
            errors.append({"file": raw, "error": "size mismatch"})
        elif _sha256(path) != str(entry.get("sha256") or ""):
            errors.append({"file": raw, "error": "SHA-256 mismatch"})
    try:
        actual = {
            path.relative_to(bundle).as_posix()
            for path in _iter_data_files(bundle)
            if path.relative_to(bundle).as_posix() != EXPORT_MANIFEST
        }
        if actual != listed:
            errors.append({"file": EXPORT_MANIFEST, "error": "file inventory does not match checksum list"})
    except PortabilityError as exc:
        errors.append({"file": EXPORT_MANIFEST, "error": str(exc)})

    workdir = _bundle_path(bundle, str(layout.get("workdir") or ""))
    if workdir is None or not workdir.is_dir():
        errors.append({"file": EXPORT_MANIFEST, "error": "workdir is missing"})
    else:
        metadata_roots = [workdir]
        bundle_runs = _bundle_path(bundle, str(layout.get("runs") or ""))
        if bundle_runs is None or not bundle_runs.is_dir():
            errors.append({"file": EXPORT_MANIFEST, "error": "configured runs directory is missing"})
        elif not _is_relative_to(bundle_runs, workdir):
            metadata_roots.append(bundle_runs)
        for metadata_root in metadata_roots:
            for path in _metadata_files(metadata_root):
                if path.name in PROVENANCE_FILENAMES:
                    continue
                try:
                    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(path.read_text())]
                except (OSError, ValueError, TypeError) as exc:
                    errors.append({"file": path.relative_to(bundle).as_posix(), "error": f"invalid metadata: {exc}"})
                    continue
                for value in values:
                    for key, raw in _strings(value):
                        if raw.startswith(PORTABLE_PREFIXES):
                            resolved, external = _resolve_bundle_uri(raw, bundle, layout)
                            counts["external_reference_paths" if external else "portable_paths"] += 1
                            if not external and (resolved is None or not resolved.exists()):
                                errors.append({"file": path.relative_to(bundle).as_posix(), "error": "managed portable path is missing or unsafe"})
                            if external and _portable_uri(raw) is None:
                                errors.append({"file": path.relative_to(bundle).as_posix(), "error": "invalid reference URI"})
                        elif raw.startswith("/") and not raw.startswith(CONTAINER_PREFIXES) and _operational_key(key) and not _provenance_key(key):
                            errors.append({"file": path.relative_to(bundle).as_posix(), "error": f"absolute operational path remains in {key}"})
    return {"destination": str(bundle), "valid": not errors, "counts": dict(sorted(counts.items())), "errors": errors}


def import_portable_workdir(bundle_path: Path, destination_app_home: Path) -> dict[str, Any]:
    bundle = Path(bundle_path).expanduser().resolve()
    requested_destination = Path(destination_app_home).expanduser()
    if requested_destination.exists() or requested_destination.is_symlink():
        raise PortabilityError(f"Import destination already exists: {requested_destination}. Choose a new app-home path; import will not modify an existing destination.")
    destination = requested_destination.resolve()
    verification = verify_portable_export(bundle)
    if not verification["valid"]:
        raise PortabilityError(f"Import refused: portable bundle verification failed ({len(verification['errors'])} errors).")
    if destination.exists() or destination.is_symlink():
        raise PortabilityError(f"Import destination already exists: {destination}. Choose a new app-home path; import will not modify an existing destination.")
    if _is_relative_to(destination, bundle) or _is_relative_to(bundle, destination):
        raise PortabilityError("Import destination and bundle cannot contain one another.")
    manifest = json.loads((bundle / EXPORT_MANIFEST).read_text())
    layout = manifest["layout"]
    source_workdir = _bundle_path(bundle, str(layout.get("workdir") or ""))
    if source_workdir is None:
        raise PortabilityError("Bundle workdir path is invalid.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.importing-{uuid4().hex}"
    try:
        staging.mkdir()
        shutil.copytree(source_workdir, staging / "workdir", copy_function=shutil.copy2)
        runs_layout = str(layout.get("runs") or "")
        if runs_layout != "workdir/runs":
            source_runs = _bundle_path(bundle, runs_layout)
            if source_runs is None or not source_runs.is_dir():
                raise PortabilityError("Configured runs directory is missing from the bundle.")
            shutil.copytree(source_runs, staging / "runs", copy_function=shutil.copy2)
        # Confirm copied files before making the destination visible.
        for entry in manifest.get("files") or []:
            raw = str(entry.get("path") or "")
            source = _bundle_path(bundle, raw)
            target = _bundle_path(staging, raw)
            if source is None or target is None or not source.is_file():
                raise PortabilityError(f"Imported file is missing: {raw}")
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if target.stat().st_size != int(entry["size"]) or _sha256(target) != entry["sha256"]:
                raise PortabilityError(f"Imported file failed checksum verification: {raw}")
        staging.rename(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    runs_relative = str(layout["runs"])
    runs_destination = destination / runs_relative
    return {
        "app_home": str(destination),
        "workdir": str(destination / "workdir"),
        "runs_dir": str(runs_destination),
        "reference_dir": "configure separately; reference_files is not included",
        "verification": verification,
    }
