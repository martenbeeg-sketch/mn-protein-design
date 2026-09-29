from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from mn_protein_design.core.gpu import docker_gpu_args
from mn_protein_design.core.jobs import read_json, write_json
from mn_protein_design.core.scheduler import apply_docker_cpu_limit
from mn_protein_design.runtime import reference_root


ESMFOLD2_IMAGE = "mn-biohub-esm:3.4.1-cu128"
BIOHUB_ESM_ROOT = reference_root() / "biohub-esm"
ESMFOLD2_HF_CACHE_VOLUME = "mn-protein-design_biohub-esm-hf-cache"
RUNTIME_DIR = Path("artifacts/raw/esmfold2_runtime")


@dataclass(frozen=True)
class ESMFold2Prediction:
    complex_path: Path
    arrays_path: Path
    complex_metadata: dict[str, Any]
    ptm: float | None
    iptm: float | None

    @property
    def result(self) -> Any:
        return _load_result(self)


def _safe_request_id(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.")
    if not text:
        raise ValueError("Each ESMFold2 request requires a non-empty request_id.")
    return text[:180]


def _relative_work_path(path: Path, run_dir: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"ESMFold2 input was not staged inside its job directory: {path}") from exc
    return f"/work/{relative.as_posix()}"


def _a3m_records(path: Path) -> list[tuple[str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with opener(path, "rt", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line, []
            elif header is not None:
                chunks.append(line)
            else:
                raise ValueError("A3M content appeared before its first FASTA header.")
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise ValueError("A3M file contains no FASTA records.")
    return records


def _a3m_match_columns(sequence: str) -> str:
    return "".join(character for character in sequence if character == "-" or character.isupper())


def _write_sanitized_a3m(source: Path, destination: Path) -> None:
    records = _a3m_records(source)
    aligned = [(header, _a3m_match_columns(sequence)) for header, sequence in records]
    query_length = len(aligned[0][1])
    lines: list[str] = []
    for header, sequence in aligned:
        sequence = sequence[:query_length].ljust(query_length, "-")
        lines.extend((header, sequence))
    destination.write_text("\n".join(lines) + "\n")


def _write_a3m_records(records: list[tuple[str, str]], destination: Path) -> None:
    destination.write_text("\n".join(part for header, sequence in records for part in (header, sequence)) + "\n")


def stage_esmfold2_msa(
    source: Path | str | None,
    expected_sequence: str,
    destination: Path,
) -> tuple[Path | None, str | None]:
    """Validate and stage an A3M without importing ESM into the app environment."""
    raw = str(source or "").strip()
    if not raw or raw.lower() == "no_msa":
        return None, None
    source_path = Path(raw).expanduser()
    if not source_path.is_file():
        return None, f"missing:{raw}"

    candidate = source_path
    note: str | None = None
    try:
        records = _a3m_records(candidate)
        match_sequences = [_a3m_match_columns(sequence) for _header, sequence in records]
        query = match_sequences[0]
        if any(len(sequence) != len(query) for sequence in match_sequences[1:]):
            raise ValueError("A3M aligned sequence lengths do not match.")
    except Exception as first_error:
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate = destination.with_name(f"{destination.stem}.sanitized.a3m")
        try:
            _write_sanitized_a3m(source_path, candidate)
            note = f"sanitized:{source_path.name}"
            records = _a3m_records(candidate)
            query = _a3m_match_columns(records[0][1])
            if any(len(_a3m_match_columns(sequence)) != len(query) for _header, sequence in records[1:]):
                raise ValueError("Sanitized A3M aligned sequence lengths do not match.")
        except Exception as sanitized_error:
            return None, f"invalid:{source_path.name}:{first_error}; sanitized_invalid:{sanitized_error}"

    normalized_query = "".join(query.upper().split())
    expected = "".join(str(expected_sequence or "").upper().split())
    if normalized_query and expected and normalized_query != expected:
        return None, f"query_mismatch:{source_path.name}"
    if candidate != source_path:
        return candidate, note
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".gz":
        _write_a3m_records(records, destination)
    else:
        shutil.copy2(source_path, destination)
    return destination, note


def _scheduled_gpu_device(run_dir: Path, gpu_device: object | None, device: str) -> str:
    if str(device).strip().lower() == "cpu":
        return "none"
    if gpu_device not in (None, ""):
        return str(gpu_device)
    metadata = read_json(run_dir / "metadata.json")
    allocation = metadata.get("resource_allocation") if isinstance(metadata.get("resource_allocation"), dict) else {}
    request = metadata.get("resource_request") if isinstance(metadata.get("resource_request"), dict) else {}
    input_payload = read_json(run_dir / "input.json")
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    selected = allocation.get("gpu_device") or request.get("gpu_device") or params.get("gpu_device")
    return str(selected or "0")


def _array_or_none(value: np.ndarray) -> np.ndarray | None:
    return value if value.size else None


def _load_prediction(run_dir: Path, row: dict[str, Any]) -> ESMFold2Prediction:
    complex_path = (run_dir / str(row["complex_path"])).resolve()
    arrays_path = (run_dir / str(row["arrays_path"])).resolve()
    for path in (complex_path, arrays_path):
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"ESMFold2 container returned an artifact outside the job directory: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"ESMFold2 container did not produce expected artifact: {path}")

    return ESMFold2Prediction(
        complex_path=complex_path,
        arrays_path=arrays_path,
        complex_metadata=dict(row["complex"]),
        ptm=row.get("ptm"),
        iptm=row.get("iptm"),
    )


def _load_result(prediction: ESMFold2Prediction) -> Any:
    metadata = prediction.complex_metadata
    with np.load(prediction.arrays_path, allow_pickle=False) as arrays:
        complex_object = SimpleNamespace(
            id=str(metadata["id"]),
            sequence=list(metadata.get("sequence") or []),
            atom_positions=arrays["atom_positions"],
            atom_elements=arrays["atom_elements"],
            atom_names=arrays["atom_names"],
            token_to_atoms=arrays["token_to_atoms"],
            chain_id=arrays["chain_id"],
            metadata=SimpleNamespace(
                entity_lookup={int(key): value for key, value in metadata.get("entity_lookup", {}).items()},
                chain_lookup={int(key): value for key, value in metadata.get("chain_lookup", {}).items()},
            ),
        )
        return SimpleNamespace(
            complex=complex_object,
            plddt=_array_or_none(arrays["plddt"]),
            pae=_array_or_none(arrays["pae"]),
            distogram=_array_or_none(arrays["distogram"]),
            pair_chains_iptm=_array_or_none(arrays["pair_chains_iptm"]),
            ptm=prediction.ptm,
            iptm=prediction.iptm,
        )


def run_esmfold2_batch(
    *,
    run_dir: Path,
    requests: list[dict[str, Any]],
    gpu_device: object | None = None,
    device: str = "auto",
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    shm_size: str = "32G",
) -> dict[str, ESMFold2Prediction]:
    """Run multiple ESMFold2 folds in the shared image, loading weights once."""
    run_dir = Path(run_dir).resolve()
    if not requests:
        return {}
    reference_root = BIOHUB_ESM_ROOT.expanduser().resolve()
    if not (reference_root / "ESMFold2").is_dir() or not (reference_root / "ESMC-6B").is_dir():
        raise FileNotFoundError(
            f"ESMFold2 and ESMC-6B reference weights are required under {reference_root}. "
            "Transfer the Biohub ESM reference directory before running this workflow."
        )
    if not re.fullmatch(r"\d+(?:[KMG]i?B?)?", str(shm_size), flags=re.IGNORECASE):
        raise ValueError(f"Invalid Docker shared-memory size: {shm_size!r}")

    runtime_dir = run_dir / RUNTIME_DIR
    input_dir = runtime_dir / "inputs"
    output_dir = runtime_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    container_script = runtime_dir / "fold_batch.py"
    shutil.copy2(Path(__file__).with_name("esmfold2_container_runner.py"), container_script)
    staged_requests: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, source_request in enumerate(requests, start=1):
        request = dict(source_request)
        request_id = _safe_request_id(request.get("request_id") or f"prediction_{index:05d}")
        if request_id in seen_ids:
            raise ValueError(f"Duplicate ESMFold2 request_id: {request_id}")
        seen_ids.add(request_id)
        request["request_id"] = request_id
        staged_sequences = []
        for chain_index, source_chain in enumerate(request.get("sequences") or [], start=1):
            chain = dict(source_chain)
            msa_source = chain.pop("msa_path", None)
            if msa_source:
                msa_path = Path(str(msa_source)).expanduser().resolve()
                if not msa_path.is_file():
                    raise FileNotFoundError(f"ESMFold2 MSA input is missing: {msa_path}")
                staged_msa = input_dir / f"{request_id}_chain_{chain_index:02d}.a3m"
                if msa_path != staged_msa.resolve():
                    shutil.copy2(msa_path, staged_msa)
                chain["msa_path"] = _relative_work_path(staged_msa, run_dir)
            staged_sequences.append(chain)
        request["sequences"] = staged_sequences
        staged_conditioning = []
        for condition_index, source_condition in enumerate(request.get("distogram_conditioning") or [], start=1):
            condition = dict(source_condition)
            source_path = condition.pop("path", None)
            array = condition.pop("array", None)
            if source_path:
                distogram_path = Path(str(source_path)).expanduser().resolve()
                if not distogram_path.is_file():
                    raise FileNotFoundError(f"ESMFold2 distogram conditioning file is missing: {distogram_path}")
                staged_path = input_dir / f"{request_id}_distogram_{condition_index:02d}.npy"
                shutil.copy2(distogram_path, staged_path)
            elif array is not None:
                staged_path = input_dir / f"{request_id}_distogram_{condition_index:02d}.npy"
                np.save(staged_path, np.asarray(array, dtype=np.float32), allow_pickle=False)
            else:
                raise ValueError("Distogram conditioning needs either a path or an array.")
            staged_conditioning.append(
                {"chain_id": str(condition["chain_id"]), "path": _relative_work_path(staged_path, run_dir)}
            )
        request["distogram_conditioning"] = staged_conditioning
        staged_requests.append(request)

    requested_device = str(device or "auto").strip().lower()
    if requested_device not in {"auto", "cuda"}:
        raise ValueError("Biohub ESMFold2 requires CUDA. Choose device=auto or cuda and allocate a GPU to this job.")
    selected_gpu = _scheduled_gpu_device(run_dir, gpu_device, requested_device)
    gpu_args = docker_gpu_args(selected_gpu)
    if not gpu_args:
        raise ValueError("Biohub ESMFold2 requires a CUDA GPU. Allocate a GPU to this job and choose device=auto or cuda.")
    container_device = "cuda"
    config = {
        "work_root": "/work",
        "reference_root": "/ref/biohub-esm",
        "output_dir": _relative_work_path(output_dir, run_dir),
        "device": container_device,
        "num_loops": int(num_loops),
        "num_sampling_steps": int(num_sampling_steps),
        "seed": int(seed),
        "requests": staged_requests,
    }
    config_path = runtime_dir / "fold_request.json"
    write_json(config_path, config)
    command = [
        "docker",
        "run",
        "--rm",
        *gpu_args,
        "--shm-size",
        str(shm_size),
        "--mount",
        f"type=bind,src={run_dir},dst=/work",
        "--mount",
        f"type=bind,src={reference_root},dst=/ref/biohub-esm,readonly",
        "--volume",
        f"{ESMFOLD2_HF_CACHE_VOLUME}:/cache/huggingface",
        ESMFOLD2_IMAGE,
        "/opt/venv/bin/python",
        _relative_work_path(container_script, run_dir),
        "--config",
        _relative_work_path(config_path, run_dir),
    ]
    command = apply_docker_cpu_limit(command, run_dir)
    write_json(
        runtime_dir / "container_command.json",
        {"command": command, "image": ESMFOLD2_IMAGE, "device": container_device},
    )
    with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.flush()
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ESMFold2 container exited with code {completed.returncode}; see {run_dir / 'stderr.log'}."
        )

    manifest_path = output_dir / "results.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"ESMFold2 container did not write its result manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest.get("results") or []:
        request_id = str(row.get("request_id") or "")
        if request_id in rows_by_id:
            raise RuntimeError(f"ESMFold2 container returned duplicate result: {request_id}")
        rows_by_id[request_id] = row
    predictions = {request_id: _load_prediction(run_dir, row) for request_id, row in rows_by_id.items()}
    unexpected = set(predictions) - seen_ids
    if unexpected:
        raise RuntimeError(f"ESMFold2 container returned unrequested predictions: {', '.join(sorted(unexpected))}")
    missing = seen_ids - set(predictions)
    if missing:
        raise RuntimeError(f"ESMFold2 container omitted prediction results: {', '.join(sorted(missing))}")
    return predictions
