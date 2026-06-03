from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path


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


def ensure_boltz_msa_for_sequence(run_dir: Path, sequence: str, label: str, raw_subdir: str = "target_msa") -> str:
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

    probe_dir = run_dir / "artifacts" / "raw" / raw_subdir / _clean_key(label)
    probe_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = probe_dir / "input.yaml"
    _write_boltz_msa_probe_yaml(yaml_path, sequence)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
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
) -> dict[str, str]:
    sequences = target_chain_sequences(target_artifact, target_chains)
    msa_paths: dict[str, str] = {}
    for chain, sequence in sequences.items():
        if not sequence:
            continue
        msa_paths[chain] = ensure_boltz_msa_for_sequence(run_dir, sequence, f"target_chain_{chain}", raw_subdir=raw_subdir)
    return msa_paths


def ensure_pxdesign_msa_dirs_for_target(
    run_dir: Path,
    target_artifact: Path,
    target_chains: list[str] | None = None,
    raw_subdir: str = "pxdesign/input/msa",
    yaml_base_dir: Path | None = None,
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
        ensure_boltz_msa_for_sequence(run_dir, sequence, f"target_chain_{chain}", raw_subdir=raw_subdir)
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
