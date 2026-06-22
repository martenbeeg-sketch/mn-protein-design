from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from mn_protein_design.core.gpu import docker_gpu_args
from mn_protein_design.runtime import runs_root


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

_A3M_SEQUENCE_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-")
BOLTZ_CACHE_DIR = Path("/mnt/db/reference_files/boltz_models")
BOLTZ_MSA_REPOSITORY_DIR = BOLTZ_CACHE_DIR / "msa_repository"
ALPHAFAST_IMAGE = "alphafast:latest"
ALPHAFAST_DB_DIR = Path("/mnt/db/reference_files/alignment")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_key(text: str, fallback: str = "target") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip()).strip("_")
    return cleaned or fallback


def pdb_residue_records_by_chain(path: Path) -> dict[str, list[tuple[int, str, str]]]:
    records: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain = line[21].strip() or "_"
        residue_key = (chain, line[22:26].strip(), line[26].strip())
        if residue_key in seen:
            continue
        seen.add(residue_key)
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        records.setdefault(chain, []).append(
            (residue_number, line[26].strip(), AA3_TO_1.get(line[17:20].strip().upper(), "X"))
        )
    return {
        chain: sorted(chain_records, key=lambda item: (item[0], item[1]))
        for chain, chain_records in records.items()
    }


def target_chain_sequences(path: Path, target_chains: list[str] | None = None) -> dict[str, str]:
    residue_records = pdb_residue_records_by_chain(path)
    requested = [chain for chain in target_chains or [] if chain in residue_records]
    chains = requested or list(residue_records)
    return {chain: "".join(record[2] for record in residue_records.get(chain, [])) for chain in chains}


def boltz_msa_paths(sequence: str, msa_repository_dir: Path = BOLTZ_MSA_REPOSITORY_DIR) -> tuple[Path, str]:
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    filename = f"{digest}.a3m"
    return msa_repository_dir / filename, f"/msa_repository/{filename}"


def _normalize_msa_query_sequence(sequence: str) -> str:
    return re.sub(r"[^A-Za-z]+", "", str(sequence or "")).upper()


def _first_a3m_query_sequence(path: Path) -> str:
    sequence_lines: list[str] = []
    in_first_record = False
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if in_first_record:
                break
            in_first_record = True
            continue
        if in_first_record:
            sequence_lines.append(line.replace("-", ""))
    return _normalize_msa_query_sequence("".join(sequence_lines))


def validate_a3m_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file does not exist"
    try:
        raw = path.read_bytes()
    except Exception as exc:
        return False, f"read failed: {exc}"
    if not raw:
        return False, "file is empty"
    if b"\x00" in raw:
        return False, "contains NUL byte(s)"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return False, f"invalid UTF-8: {exc}"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, "contains no non-empty lines"
    if not lines[0].startswith(">"):
        return False, "first non-empty line is not a FASTA header"
    saw_sequence = False
    for line in lines:
        if line.startswith(">"):
            continue
        saw_sequence = True
        if any(char not in _A3M_SEQUENCE_ALLOWED for char in line):
            return False, "contains invalid sequence characters"
    if not saw_sequence:
        return False, "contains headers only and no sequence"
    return True, ""


def find_cached_msa_for_sequence(
    sequence: str,
    msa_repository_dir: Path = BOLTZ_MSA_REPOSITORY_DIR,
) -> tuple[Path | None, str]:
    """Find a valid cached A3M for an exact protein sequence.

    The canonical cache path is the SHA256 of the sequence, but older app runs may
    have written valid A3Ms with job-local or descriptive filenames. Scanning by
    the first A3M/query sequence lets new runs reuse those files without tying MSA
    discovery to a particular target-preparation run ID.
    """
    cleaned_sequence = "".join(str(sequence or "").split()).upper()
    if not cleaned_sequence:
        return None, "empty sequence"
    host_path, _container_path = boltz_msa_paths(cleaned_sequence, msa_repository_dir=msa_repository_dir)
    valid, reason = validate_a3m_file(host_path)
    if valid:
        return host_path, "sequence_hash"
    if host_path.exists():
        repaired = _repair_a3m_file(host_path)
        if repaired:
            return host_path, "sequence_hash_repaired"
    if not Path(msa_repository_dir).exists():
        return None, f"repository missing; hash path: {reason}"

    expected_query = _normalize_msa_query_sequence(cleaned_sequence)
    for candidate in sorted(Path(msa_repository_dir).glob("**/*.a3m")):
        if candidate == host_path:
            continue
        valid, _candidate_reason = validate_a3m_file(candidate)
        if not valid and not _repair_a3m_file(candidate):
            continue
        try:
            if _first_a3m_query_sequence(candidate) == expected_query:
                return candidate, "sequence_scan"
        except Exception:
            continue
    return None, reason or "no sequence-matched A3M in repository"


def _validate_a3m_payload(raw: bytes) -> tuple[bool, str]:
    temp = Path("/tmp") / f"mn_protein_design_a3m_check_{hashlib.sha256(raw).hexdigest()}.a3m"
    try:
        temp.write_bytes(raw)
        return validate_a3m_file(temp)
    finally:
        temp.unlink(missing_ok=True)


def _repair_a3m_file(path: Path) -> bool:
    if not path.exists():
        return False
    cleaned = path.read_bytes().rstrip(b"\x00").replace(b"\r\n", b"\n")
    valid, _reason = _validate_a3m_payload(cleaned)
    if not valid:
        return False
    if cleaned != path.read_bytes():
        path.write_bytes(cleaned)
    return True


def _first_valid_a3m(root: Path) -> Path | None:
    for candidate in sorted(root.glob("**/*.a3m"), key=lambda path: ("processed" not in str(path), len(str(path)))):
        valid, _reason = validate_a3m_file(candidate)
        if valid:
            return candidate
        if _repair_a3m_file(candidate):
            return candidate
    return None


def _write_boltz_msa_probe_yaml(path: Path, sequence: str) -> None:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "sequences:",
                "  - protein:",
                "      id: A",
                f"      sequence: {sequence}",
                "",
            ]
        )
    )


def _repo_and_runs_mounts() -> list[str]:
    mounts = ["-v", f"{REPO_ROOT}:{REPO_ROOT}"]
    try:
        run_root = runs_root().resolve()
        repo_root = REPO_ROOT.resolve()
    except Exception:
        return mounts
    try:
        run_root.relative_to(repo_root)
    except ValueError:
        mounts.extend(["-v", f"{run_root}:{run_root}"])
    return mounts


def _write_alphafast_msa_input(path: Path, sequence: str, label: str) -> None:
    path.write_text(
        json.dumps(
            {
                "dialect": "alphafold3",
                "version": 4,
                "name": _clean_key(label),
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": sequence,
                            "templates": [],
                        }
                    }
                ],
                "modelSeeds": [1],
            },
            indent=2,
        )
    )


def _extract_alphafast_unpaired_msa(output_dir: Path, sequence: str) -> str:
    for data_json in sorted(output_dir.glob("*/*_data.json")):
        try:
            payload = json.loads(data_json.read_text())
        except Exception:
            continue
        for entry in payload.get("sequences") or []:
            protein = entry.get("protein") if isinstance(entry, dict) else None
            if not isinstance(protein, dict):
                continue
            if str(protein.get("sequence") or "").strip() != sequence:
                continue
            msa_text = str(protein.get("unpairedMsa") or "").strip()
            if msa_text:
                return msa_text + "\n"
    return ""


def _ensure_alphafast_mmseqs_msa_for_sequence(
    run_dir: Path,
    sequence: str,
    label: str,
    raw_subdir: str,
    gpu_device: object,
    image: str = ALPHAFAST_IMAGE,
    db_dir: Path = ALPHAFAST_DB_DIR,
) -> str:
    if not db_dir.exists():
        raise RuntimeError(f"AlphaFast database directory does not exist: {db_dir}")
    if not (db_dir / "mmseqs").exists():
        raise RuntimeError(f"AlphaFast database directory must contain an mmseqs subdirectory: {db_dir / 'mmseqs'}")
    probe_dir = run_dir / "artifacts" / "raw" / raw_subdir / _clean_key(label)
    input_dir = probe_dir / "input"
    output_dir = probe_dir / "alphafast_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_alphafast_msa_input(input_dir / f"{_clean_key(label)}.json", sequence, label)
    command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
        *_repo_and_runs_mounts(),
        "-v",
        f"{db_dir}:/data/public_databases:ro",
        "-v",
        f"{db_dir / 'mmseqs'}:/data/mmseqs_databases:ro",
        "-w",
        "/app/alphafold",
        image,
        "python",
        "/app/alphafold/run_data_pipeline.py",
        f"--input_dir={input_dir}",
        f"--output_dir={output_dir}",
        "--db_dir=/data/public_databases",
        "--mmseqs_db_dir=/data/mmseqs_databases",
        "--use_mmseqs_gpu",
        "--batch_size=1",
    ]
    with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.write(f"Local AlphaFast/MMseqs MSA cache miss for {label}\n")
        stdout.flush()
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"AlphaFast/MMseqs MSA preparation failed for {label} with return code {completed.returncode}.")
    msa_text = _extract_alphafast_unpaired_msa(output_dir, sequence)
    if not msa_text:
        raise RuntimeError(f"AlphaFast/MMseqs did not produce an unpaired MSA for {label}.")
    return msa_text


def ensure_boltz_msa_for_sequence(
    run_dir: Path,
    sequence: str,
    label: str,
    raw_subdir: str = "target_msa",
    gpu_device: object = "0",
    msa_source: str = "alphafast_mmseqs_gpu",
) -> str:
    host_path, container_path = boltz_msa_paths(sequence)
    host_path.parent.mkdir(parents=True, exist_ok=True)
    valid, reason = validate_a3m_file(host_path)
    if not valid and host_path.exists():
        if _repair_a3m_file(host_path):
            valid, reason = validate_a3m_file(host_path)
    if valid:
        with (run_dir / "stdout.log").open("a") as stdout:
            stdout.write(f"MSA cache hit for {label}: {host_path}\n")
        return container_path

    if str(msa_source or "").strip().lower() in {"alphafast_mmseqs_gpu", "local_alphafast_mmseqs_gpu", "local"}:
        payload = _ensure_alphafast_mmseqs_msa_for_sequence(
            run_dir,
            sequence,
            label,
            raw_subdir=raw_subdir,
            gpu_device=gpu_device,
        ).encode("utf-8").rstrip(b"\x00").replace(b"\r\n", b"\n")
        valid_payload, payload_reason = _validate_a3m_payload(payload)
        if not valid_payload:
            raise RuntimeError(f"AlphaFast/MMseqs produced invalid A3M for {label}: {payload_reason}.")
        host_path.write_bytes(payload)
        valid, reason = validate_a3m_file(host_path)
        if not valid:
            raise RuntimeError(f"Cached AlphaFast/MMseqs MSA is invalid for {label}: {reason}.")
        with (run_dir / "stdout.log").open("a") as stdout:
            stdout.write(f"MSA cached for {label} from local AlphaFast/MMseqs GPU: {host_path}\n")
        return container_path

    if str(msa_source or "").strip().lower() not in {"boltz_msa_server", "boltz_server", "remote_boltz_msa_server"}:
        raise RuntimeError(f"Unknown target MSA source: {msa_source}")

    probe_dir = run_dir / "artifacts" / "raw" / raw_subdir / _clean_key(label)
    probe_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = probe_dir / "input.yaml"
    _write_boltz_msa_probe_yaml(yaml_path, sequence)
    command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
        "-v",
        f"{probe_dir}:/work",
        "-v",
        f"{BOLTZ_CACHE_DIR}:/cache",
        "-v",
        f"{BOLTZ_MSA_REPOSITORY_DIR}:/msa_repository",
        "-e",
        "BOLTZ_CACHE=/cache",
        "--ipc=host",
        "--shm-size=48G",
        "ovoex-boltz2",
        "predict",
        "/work/input.yaml",
        "--out_dir",
        "/work",
        "--sampling_steps",
        "200",
        "--recycling_steps",
        "3",
        "--diffusion_samples",
        "1",
        "--accelerator",
        "gpu",
        "--override",
        "--use_msa_server",
    ]
    with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.write(f"MSA cache miss for {label}: {host_path} ({reason})\n")
        stdout.flush()
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Boltz2 MSA preflight failed for {label} with return code {completed.returncode}.")
    generated = _first_valid_a3m(probe_dir)
    if generated is None:
        raise RuntimeError(f"Boltz2 MSA preflight did not produce a valid A3M for {label}.")
    payload = generated.read_bytes().rstrip(b"\x00").replace(b"\r\n", b"\n")
    valid_payload, payload_reason = _validate_a3m_payload(payload)
    if not valid_payload:
        raise RuntimeError(f"Boltz2 MSA preflight produced invalid A3M for {label}: {payload_reason}.")
    host_path.write_bytes(payload)
    valid, reason = validate_a3m_file(host_path)
    if not valid:
        raise RuntimeError(f"Cached Boltz2 MSA is invalid for {label}: {reason}.")
    with (run_dir / "stdout.log").open("a") as stdout:
        stdout.write(f"MSA cached for {label}: {host_path}\n")
    return container_path


def ensure_boltz_msas_for_target(
    run_dir: Path,
    target_artifact: Path,
    target_chains: list[str] | None = None,
    raw_subdir: str = "target_msa",
    gpu_device: object = "0",
) -> dict[str, str]:
    sequences = target_chain_sequences(target_artifact, target_chains)
    msa_paths: dict[str, str] = {}
    for chain, sequence in sequences.items():
        if not sequence:
            continue
        msa_paths[chain] = ensure_boltz_msa_for_sequence(
            run_dir,
            sequence,
            f"target_chain_{chain}",
            raw_subdir=raw_subdir,
            gpu_device=gpu_device,
        )
    return msa_paths


def ensure_pxdesign_msa_dirs_for_target(
    run_dir: Path,
    target_artifact: Path,
    target_chains: list[str] | None = None,
    raw_subdir: str = "pxdesign/input/msa",
    yaml_base_dir: Path | None = None,
    gpu_device: object = "0",
) -> dict[str, str]:
    """Create PXDesign-compatible MSA directories from the shared A3M cache.

    PXDesign/Protenix expects each chain MSA as a directory containing
    ``pairing.a3m`` and ``non_pairing.a3m``. The app keeps a single
    sequence-hashed A3M archive for reuse across workflows, then materializes
    per-run directories next to the PXDesign input YAML.
    """

    sequences = target_chain_sequences(target_artifact, target_chains)
    msa_dirs: dict[str, str] = {}
    yaml_base_dir = yaml_base_dir or run_dir
    for chain, sequence in sequences.items():
        if not sequence:
            continue
        ensure_boltz_msa_for_sequence(
            run_dir,
            sequence,
            f"target_chain_{chain}",
            raw_subdir=raw_subdir,
            gpu_device=gpu_device,
        )
        host_msa, _container_msa = boltz_msa_paths(sequence)
        valid, reason = validate_a3m_file(host_msa)
        if not valid:
            raise RuntimeError(f"Cached target MSA for chain {chain} is invalid: {reason}.")
        chain_dir = run_dir / "artifacts" / "raw" / raw_subdir / _clean_key(chain, fallback="chain")
        chain_dir.mkdir(parents=True, exist_ok=True)
        for filename in ["pairing.a3m", "non_pairing.a3m"]:
            shutil.copyfile(host_msa, chain_dir / filename)
        try:
            msa_dirs[chain] = str(chain_dir.relative_to(yaml_base_dir))
        except ValueError:
            msa_dirs[chain] = str(chain_dir)
    return msa_dirs
