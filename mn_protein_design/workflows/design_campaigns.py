from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, read_candidates, write_candidates
from mn_protein_design.core.jobs import (
    create_job,
    finish_job,
    mark_internal_job,
    read_json,
    update_status,
    write_json,
)
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows.analysis import _run_ipsae
from mn_protein_design.workflows.design import (
    default_target_contig,
    run_bindcraft,
    run_boltzgen,
    run_genie3,
    run_proteina_complexa,
    run_protpardelle_1c,
    run_pxdesign,
    run_rfdiffusion3_foundry,
    run_rfdiffusion_classic,
)
from mn_protein_design.workflows.esm_binder import run_esmfold2_native_binder_design
from mn_protein_design.workflows.refolding import run_af2_initial_guess_complex_refolding
from mn_protein_design.workflows.sequence_design import run_ligandmpnn_sequence_design


DESIGN_CAMPAIGN_GROUP = "design-campaign"
ENGINE_ORDER = (
    "template_redesign",
    "rfdiffusion_classic",
    "bindcraft",
    "rfdiffusion3_foundry",
    "boltzgen",
    "pxdesign",
    "genie3",
    "esmfold2_binder_design",
    "protpardelle_1c",
    "proteina_complexa",
)
ENGINE_LABELS = {
    "template_redesign": "Template redesign",
    "rfdiffusion_classic": "RFdiffusion classic",
    "bindcraft": "BindCraft",
    "rfdiffusion3_foundry": "RFdiffusion3 / Foundry",
    "boltzgen": "BoltzGen",
    "pxdesign": "PXDesign",
    "genie3": "Genie3",
    "esmfold2_binder_design": "ESMFold2 binder design",
    "protpardelle_1c": "Protpardelle-1c",
    "proteina_complexa": "Proteina-Complexa",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
PYROSETTA_METRICS_SCRIPT = REPO_ROOT / "tools_to_implement" / "de_novo_binder_scoring" / "scripts" / "compute_rosetta_metrics.py"
PYROSETTA_METRICS_WORKDIR = REPO_ROOT / "tools_to_implement" / "de_novo_binder_scoring"
PYROSETTA_METRICS_IMAGE = "ovo-bindcraft:latest"
COMMON_AF2_THRESHOLDS = {
    "model_1_binder_plddt": (80.0, "higher"),
    "model_2_binder_plddt": (80.0, "higher"),
    "model_1_ptm": (0.55, "higher"),
    "model_2_ptm": (0.55, "higher"),
    "model_1_iptm": (0.50, "higher"),
    "model_2_iptm": (0.50, "higher"),
    "model_1_ipae": (0.35 * 31.0, "lower"),
    "model_2_ipae": (0.35 * 31.0, "lower"),
    "model_1_binder_only_plddt": (80.0, "higher"),
    "model_2_binder_only_plddt": (80.0, "higher"),
    "model_1_binder_rmsd": (3.5, "lower"),
    "model_2_binder_rmsd": (3.5, "lower"),
}
COMMON_ROSETTA_THRESHOLDS = {
    "rosetta_binder_score": (0.0, "lower"),
    "rosetta_surface_hydrophobicity": (0.35, "lower"),
    "rosetta_interface_sc": (0.55, "higher"),
    "rosetta_interface_dG": (0.0, "lower"),
    "rosetta_interface_dSASA": (1.0, "higher"),
    "rosetta_interface_nres_binder": (7.0, "higher"),
    "rosetta_interface_interface_hbonds": (3.0, "higher"),
    "rosetta_interface_unsat_hbonds": (4.0, "lower"),
    "rosetta_interface_aa_LYS": (3.0, "lower"),
    "rosetta_interface_aa_MET": (3.0, "lower"),
}
COMMON_STRUCTURE_THRESHOLDS = {
    "average_binder_loop_percent": (90.0, "lower"),
    "model_1_binder_loop_percent": (90.0, "lower"),
    "model_2_binder_loop_percent": (90.0, "lower"),
    "average_hotspot_rmsd": (6.0, "lower"),
    "model_1_hotspot_rmsd": (6.0, "lower"),
    "model_2_hotspot_rmsd": (6.0, "lower"),
}
CANDIDATE_POOL_COVERAGE = {
    "template_redesign": "uploaded target-binder complex before sequence refinement",
    "rfdiffusion_classic": "normalized outputs retained by the app pipeline",
    "bindcraft": "no-filter completed designs when common validation is enabled",
    "rfdiffusion3_foundry": "RF3 native survivors plus RFdiffusion3/Foundry-MPNN prefilter structures",
    "boltzgen": "final ranked designs plus intermediate generated BoltzGen structures",
    "pxdesign": "PXDesign summary rows plus raw inference prediction CIFs",
    "genie3": "native evaluation rows plus generated Genie3 PDBs",
    "esmfold2_binder_design": "all completed optimization starts",
    "protpardelle_1c": "native ESMFold/MPNN rows plus raw scaffold samples",
    "proteina_complexa": "ranked evaluation rows plus generated inference complexes",
}
REFINEMENT_MODEL_TYPES = {
    "ProteinMPNN": "protein_mpnn",
    "LigandMPNN": "ligand_mpnn",
    "Soluble ProteinMPNN": "soluble_mpnn",
    "protein_mpnn": "protein_mpnn",
    "ligand_mpnn": "ligand_mpnn",
    "soluble_mpnn": "soluble_mpnn",
}
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
    "MSE": "M",
}


def _number(value: object, default: float = math.inf) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _first_metric(metrics: dict[str, Any], names: tuple[str, ...], default: float = math.inf) -> float:
    for name in names:
        if name in metrics and metrics[name] not in {None, ""}:
            return _number(metrics[name], default)
    return default


def _native_pass(candidate: dict[str, Any]) -> bool | None:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    for name in ("native_pass_filters", "pass_filters", "passes_filters", "passed_filters"):
        value = metrics.get(name)
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"true", "pass", "passed", "1", "yes"}:
            return True
        if str(value).strip().lower() in {"false", "fail", "failed", "0", "no"}:
            return False
    return None


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    passed = _native_pass(candidate)
    native_rank = _first_metric(
        metrics,
        (
            "native_final_rank",
            "native_design_rank",
            "esmfold2_screening_rank",
            "design_rank",
            "rank",
            "native_design_index",
        ),
    )
    confidence = _first_metric(
        metrics,
        ("analysis_score", "iptm", "ipTM", "confidence", "binder_plddt", "plddt", "ipsae", "ipSAE"),
        default=-math.inf,
    )
    interface_error = _first_metric(
        metrics,
        ("ipae", "iPAE", "interface_pae", "binder_rmsd"),
    )
    return (
        0.0 if passed is True else 1.0 if passed is None else 2.0,
        native_rank,
        -confidence,
        interface_error,
    )


def _absolute_candidate_paths(candidate: dict[str, Any], child_run: Path) -> dict[str, Any]:
    normalized = dict(candidate)
    for key in ("target_pdb", "complex_pdb", "binder_pdb"):
        value = str(normalized.get(key) or "").strip()
        if value and not Path(value).is_absolute():
            normalized[key] = str((child_run / value).resolve())
    raw_metadata = dict(normalized.get("raw_metadata") or {})
    for key, value in list(raw_metadata.items()):
        if not key.endswith(("_path", "_pdb", "_json", "_npz")):
            continue
        text = str(value or "").strip()
        if text and not Path(text).is_absolute() and (child_run / text).exists():
            raw_metadata[key] = str((child_run / text).resolve())
    normalized["raw_metadata"] = raw_metadata
    return normalized


def _bindcraft_design_parameter_family(child_run: Path) -> str:
    settings_path = child_run / "artifacts" / "raw" / "bindcraft" / "settings_advanced.json"
    settings = read_json(settings_path) if settings_path.exists() else {}
    if "use_multimer_design" in settings:
        return "multimer_v3" if bool(settings.get("use_multimer_design")) else "ptm"
    payload = read_json(child_run / "input.json")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    settings_name = str(params.get("advanced_settings_file") or "").lower()
    if settings_name:
        return "multimer_v3" if "multimer" in settings_name else "ptm"
    return "multimer_v3"


def _candidate_design_parameter_family(
    candidate: dict[str, Any],
    *,
    engine: str,
    child_run: Path,
) -> str:
    metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    explicit = str(
        metadata.get("design_af2_parameter_family")
        or metrics.get("design_af2_parameter_family")
        or ""
    ).strip().lower()
    if explicit in {"multimer", "multimer_v3", "af2_multimer", "af2_multimer_v3"}:
        return "multimer_v3"
    if explicit in {"ptm", "af2_ptm", "monomer_ptm"}:
        return "ptm"
    if engine == "bindcraft":
        return _bindcraft_design_parameter_family(child_run)
    return "none"


def _validation_parameter_family(design_family: str) -> str:
    _ = design_family
    return "multimer_v3"


def _all_child_candidates(child_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for child in child_runs:
        engine = str(child["engine"])
        child_run = Path(str(child["run_dir"]))
        for index, candidate in enumerate(read_candidates(child_run), start=1):
            candidate = _absolute_candidate_paths(candidate, child_run)
            source_id = str(candidate.get("candidate_id") or f"candidate-{index}")
            metadata = dict(candidate.get("raw_metadata") or {})
            design_family = _candidate_design_parameter_family(
                candidate,
                engine=engine,
                child_run=child_run,
            )
            metadata.update(
                {
                    "design_campaign_engine": engine,
                    "design_af2_parameter_family": design_family,
                    "common_validation_af2_parameter_family": _validation_parameter_family(
                        design_family
                    ),
                    "native_source_run_dir": str(child_run),
                    "native_source_run_id": child_run.name,
                    "native_source_candidate_id": source_id,
                    "native_pass": _native_pass(candidate),
                    "candidate_pool_coverage": CANDIDATE_POOL_COVERAGE.get(engine, "native emitted candidates"),
                }
            )
            if not metadata.get("design_reference_pdb") and candidate.get("complex_pdb"):
                metadata["design_reference_pdb"] = candidate.get("complex_pdb")
                metadata["design_reference_kind"] = "native_emitted_complex"
            candidate.update(
                {
                    "candidate_id": f"{engine}__{source_id}",
                    "raw_metadata": metadata,
                }
            )
            candidates.append(candidate)
    return candidates


def _write_candidate_jsonl(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(candidate, sort_keys=True) + "\n" for candidate in candidates))


def _threshold_failures(
    metrics: dict[str, Any],
    thresholds: dict[str, tuple[float, str]],
    *,
    missing_is_failure: bool = True,
) -> list[str]:
    failures: list[str] = []
    for metric, (threshold, direction) in thresholds.items():
        value = metrics.get(metric)
        if value in {None, ""}:
            if missing_is_failure:
                failures.append(f"{metric} missing")
            continue
        number = _number(value)
        if not math.isfinite(number):
            failures.append(f"{metric} invalid")
        elif direction == "higher" and number < threshold:
            failures.append(f"{metric} < {threshold:g}")
        elif direction == "lower" and number > threshold:
            failures.append(f"{metric} > {threshold:g}")
    return failures


def _docker_mounts_for_path(path: Path) -> list[str]:
    mounts = ["-v", f"{REPO_ROOT}:{REPO_ROOT}"]
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        mounts.extend(["-v", f"{runs_root().resolve()}:{runs_root().resolve()}"])
    return mounts


def _pdb_chain_sequences(path: Path, chains: list[str]) -> dict[str, str]:
    keep = set(chains)
    residues: dict[str, list[tuple[tuple[int, str], str]]] = {}
    seen: set[tuple[str, int, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        if keep and chain not in keep:
            continue
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        key = (chain, residue_number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        aa = AA3_TO_1.get(line[17:20].strip().upper(), "X")
        residues.setdefault(chain, []).append(((residue_number, insertion_code), aa))
    return {
        chain: "".join(aa for _, aa in sorted(values, key=lambda item: item[0]))
        for chain, values in residues.items()
    }


def _pdb_residue_tags(path: Path, chains: list[str]) -> list[str]:
    keep = set(chains)
    tags: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        if keep and chain not in keep:
            continue
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        key = (chain, residue_number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        tags.append(f"{chain}{residue_number}{insertion_code}")
    return tags


def _parse_residue_tags(text: object, *, default_chain: str = "") -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for token in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        token = token.strip().replace(":", "")
        if not token:
            continue
        chain = token[0].upper() if token[0].isalpha() else default_chain
        residue = token[1:] if token and token[0].isalpha() else token
        residue = residue.strip()
        if not chain or not residue:
            continue
        tag = f"{chain}{residue}"
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def _create_template_redesign_source(
    run_dir: Path,
    *,
    target_pdb: Path,
    template_config: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(template_config.get("enabled")):
        return None
    source_pdb = Path(str(template_config.get("complex_pdb") or "")).expanduser()
    if not source_pdb.exists():
        raise FileNotFoundError(f"Template complex PDB does not exist: {source_pdb}")
    target_chains = [str(chain) for chain in template_config.get("target_chains") or []]
    binder_chains = [str(chain) for chain in template_config.get("binder_chains") or []]
    if not target_chains:
        raise ValueError("Template redesign requires at least one template target chain.")
    if not binder_chains:
        raise ValueError("Template redesign requires at least one template binder chain.")

    source_dir = run_dir / "artifacts" / "design_campaign" / "template_redesign" / "source_run"
    complex_dir = source_dir / "artifacts" / "template_redesign"
    complex_dir.mkdir(parents=True, exist_ok=True)
    copied_complex = complex_dir / "template_complex.pdb"
    shutil.copy2(source_pdb, copied_complex)
    sequences = _pdb_chain_sequences(copied_complex, binder_chains)
    binder_sequence = ":".join(sequences.get(chain, "") for chain in binder_chains)
    residue_tags = _pdb_residue_tags(copied_complex, binder_chains)
    candidate = {
        "candidate_id": "template_redesign_00001",
        "stage": STAGE_COMPLEX_REFOLDING,
        "source_tool": "template_redesign",
        "target_pdb": str(target_pdb),
        "complex_pdb": str(copied_complex.relative_to(source_dir)),
        "binder_sequence": binder_sequence,
        "target_chains": target_chains,
        "binder_chains": binder_chains,
        "binder_length": str(sum(len(sequence) for sequence in sequences.values())),
        "metrics": {
            "native_final_rank": 1,
            "native_design_rank": 1,
            "native_pass_filters": True,
            "pass_filters": True,
            "candidate_pool_level": "template_input",
            "template_binder_residue_count": len(residue_tags),
        },
        "raw_metadata": {
            "result_kind": "template_redesign",
            "template_source_pdb": str(source_pdb),
            "template_mode": str(template_config.get("mode") or "interface"),
            "template_locked_residues": str(template_config.get("locked_residues") or ""),
            "template_unlocked_residues": str(template_config.get("unlocked_residues") or ""),
            "candidate_pool_level": "template_input",
            "design_reference_pdb": str(copied_complex),
            "design_reference_kind": "template_complex",
            "input_binder_chains": binder_chains,
            "input_target_chains": target_chains,
            "template_binder_residues": residue_tags,
        },
    }
    write_candidates(source_dir, "template_redesign", [candidate])
    write_json(
        source_dir / "result.json",
        {
            "success": True,
            "metrics": {"candidate_count": 1, "engine": "template_redesign"},
            "outputs": {"candidates": "artifacts/normalized_candidates/candidates.jsonl"},
        },
    )
    return {
        "engine": "template_redesign",
        "run_dir": str(source_dir),
        "run_id": source_dir.name,
        "status": "completed",
        "error": "",
    }


def _run_common_pyrosetta(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    nprocs: int,
) -> tuple[dict[str, dict[str, Any]], Path, int]:
    scoring_dir = run_dir / "artifacts" / "design_campaign" / "common_validation" / "pyrosetta_inputs"
    relaxed_dir = run_dir / "artifacts" / "design_campaign" / "common_validation" / "pyrosetta_relaxed"
    output_csv = run_dir / "artifacts" / "design_campaign" / "common_validation" / "pyrosetta_metrics.csv"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        structure = Path(str(candidate.get("complex_pdb") or ""))
        if not structure.exists() or structure.suffix.lower() != ".pdb":
            continue
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(candidate["candidate_id"]))
        shutil.copy2(structure, scoring_dir / f"{safe_id}.pdb")
    if not list(scoring_dir.glob("*.pdb")):
        return {}, output_csv, 0
    command = [
        "docker",
        "run",
        "--rm",
        *_docker_mounts_for_path(run_dir),
        "-w",
        str(PYROSETTA_METRICS_WORKDIR),
        PYROSETTA_METRICS_IMAGE,
        "python",
        str(PYROSETTA_METRICS_SCRIPT),
        "--folder",
        f"common:{scoring_dir}",
        "--out-csv",
        str(output_csv),
        "--relaxed-dir",
        str(relaxed_dir),
        "--binder-chain",
        "A",
        "--target-chain",
        "B",
        "--nprocs",
        str(max(1, int(nprocs))),
        "--dalphaball-path",
        str(PYROSETTA_METRICS_WORKDIR / "functions" / "DAlphaBall.gcc"),
    ]
    with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.flush()
        rc = subprocess.run(command, stdout=stdout, stderr=stderr, check=False).returncode
    rows: dict[str, dict[str, Any]] = {}
    if output_csv.exists():
        with output_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                binder_id = str(row.get("binder_id") or "")
                metrics = {
                    key.removeprefix("common_"): value
                    for key, value in row.items()
                    if key != "binder_id" and value not in {None, ""}
                }
                rows[binder_id] = metrics
    return rows, output_csv, int(rc)


def _secondary_structure_metrics(
    path: Path,
    *,
    binder_chain: str = "A",
    target_chains: list[str] | None = None,
) -> dict[str, float]:
    from Bio.PDB import DSSP, PDBParser

    lines = [
        "HEADER    MN PROTEIN DESIGN DSSP INPUT",
        "CRYST1  500.000  500.000  500.000  90.00  90.00  90.00 P 1           1",
    ]
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "HETATM", "TER   ")):
            lines.append(line)
    lines.append("END")
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        structure = PDBParser(QUIET=True).get_structure(
            "common_validation", handle.name
        )
        model = next(structure.get_models())
        dssp = DSSP(model, handle.name, dssp="mkdssp")
    targets = target_chains or [chain.id for chain in model if chain.id != binder_chain]
    interface = set(
        _binder_interface_residues(path, [binder_chain], targets, cutoff=4.0)
    )
    binder_counts = {"helix": 0, "sheet": 0, "loop": 0}
    interface_counts = {"helix": 0, "sheet": 0, "loop": 0}
    for key in dssp.keys():
        chain, residue_id = key
        if chain != binder_chain:
            continue
        code = str(dssp[key][2])
        kind = "helix" if code in {"H", "G", "I"} else "sheet" if code == "E" else "loop"
        binder_counts[kind] += 1
        residue_tag = f"{chain}{residue_id[1]}{str(residue_id[2]).strip()}"
        if residue_tag in interface:
            interface_counts[kind] += 1

    def percentages(counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values())
        return {
            kind: round(100.0 * count / total, 2) if total else 0.0
            for kind, count in counts.items()
        }

    binder = percentages(binder_counts)
    interface_ss = percentages(interface_counts)
    return {
        "binder_helix_percent": binder["helix"],
        "binder_betasheet_percent": binder["sheet"],
        "binder_loop_percent": binder["loop"],
        "interface_helix_percent": interface_ss["helix"],
        "interface_betasheet_percent": interface_ss["sheet"],
        "interface_loop_percent": interface_ss["loop"],
    }


def _chain_ca_coordinates(path: Path, chains: list[str]) -> list[tuple[float, float, float]]:
    keep = set(chains)
    coordinates: list[tuple[float, float, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if (line[21].strip() or "_") not in keep:
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        try:
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
        except ValueError:
            continue
    return coordinates


def _unaligned_ca_rmsd(
    reference: Path,
    prediction: Path,
    *,
    reference_chains: list[str],
    prediction_chains: list[str],
) -> float | None:
    reference_coords = _chain_ca_coordinates(reference, reference_chains)
    prediction_coords = _chain_ca_coordinates(prediction, prediction_chains)
    count = min(len(reference_coords), len(prediction_coords))
    if not count:
        return None
    squared = sum(
        sum(
            (reference_coords[index][axis] - prediction_coords[index][axis]) ** 2
            for axis in range(3)
        )
        for index in range(count)
    )
    return math.sqrt(squared / count)


def _common_prediction_model_paths(candidate: dict[str, Any]) -> list[Path]:
    raw = candidate.get("raw_metadata") or {}
    prediction_dir = Path(str(raw.get("prediction_dir") or ""))
    if not prediction_dir.exists():
        complex_path = Path(str(candidate.get("complex_pdb") or ""))
        prediction_dir = complex_path.parent
    safe_id = str((candidate.get("metrics") or {}).get("id") or "")
    paths: list[Path] = []
    for model_number in (1, 2):
        matches = sorted(
            path
            for path in prediction_dir.glob(f"{safe_id}_*_model{model_number}.pdb")
            if "_binder_model" not in path.name
        )
        if not matches:
            matches = sorted(
                path
                for path in prediction_dir.glob(f"*model{model_number}.pdb")
                if safe_id in path.name and "_binder_model" not in path.name
            )
        if matches:
            paths.append(matches[0])
    return paths


def _add_bindcraft_structure_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(candidate.get("metrics") or {})
    raw = candidate.get("raw_metadata") or {}
    reference_text = str(raw.get("design_reference_pdb") or "").strip()
    reference_path = Path(reference_text) if reference_text else None
    reference_chains = [
        str(chain)
        for chain in raw.get("input_binder_chains") or candidate.get("binder_chains") or ["A"]
    ]
    model_paths = _common_prediction_model_paths(candidate)
    structure_errors: list[str] = []
    for model_number, model_path in enumerate(model_paths, start=1):
        try:
            ss_metrics = _secondary_structure_metrics(
                model_path,
                binder_chain="A",
                target_chains=["B"],
            )
            metrics.update(
                {
                    f"model_{model_number}_{key}": value
                    for key, value in ss_metrics.items()
                }
            )
        except Exception as exc:
            structure_errors.append(
                f"model {model_number} DSSP: {type(exc).__name__}: {exc}"
            )
        if reference_path is not None and reference_path.exists():
            rmsd = _unaligned_ca_rmsd(
                reference_path,
                model_path,
                reference_chains=reference_chains,
                prediction_chains=["A"],
            )
            metrics[f"model_{model_number}_hotspot_rmsd"] = rmsd

    averaged_names = (
        "binder_helix_percent",
        "binder_betasheet_percent",
        "binder_loop_percent",
        "interface_helix_percent",
        "interface_betasheet_percent",
        "interface_loop_percent",
        "hotspot_rmsd",
    )
    for metric_name in averaged_names:
        values = [
            metrics.get(f"model_{model_number}_{metric_name}")
            for model_number in range(1, len(model_paths) + 1)
        ]
        numeric = [float(value) for value in values if value is not None]
        metrics[f"average_{metric_name}"] = (
            sum(numeric) / len(numeric) if numeric else None
        )
    metrics.update(
        {
            "bindcraft_structure_metric_model_count": len(model_paths),
            "hotspot_rmsd_reference_available": bool(
                reference_path is not None and reference_path.exists()
            ),
            "hotspot_rmsd_reference_kind": str(raw.get("design_reference_kind") or ""),
            "bindcraft_structure_metric_errors": "; ".join(structure_errors),
        }
    )
    candidate["metrics"] = metrics
    return candidate


def run_common_bindcraft_validation(
    run_dir: Path,
    child_runs: list[dict[str, Any]],
    *,
    gpu_device: str,
    num_recycles: int,
    min_ipsae: float,
    pyrosetta_nprocs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    native_candidates = _all_child_candidates(child_runs)
    pool_level_counts: dict[str, int] = {}
    for candidate in native_candidates:
        pool_level = str(
            (candidate.get("metrics") or {}).get("candidate_pool_level")
            or (candidate.get("raw_metadata") or {}).get("candidate_pool_level")
            or "unspecified"
        )
        pool_level_counts[pool_level] = pool_level_counts.get(pool_level, 0) + 1
    validation_dir = run_dir / "artifacts" / "design_campaign" / "common_validation"
    source_jsonl = validation_dir / "native_candidates.jsonl"
    _write_candidate_jsonl(source_jsonl, native_candidates)
    if not native_candidates:
        return [], {"native_candidate_count": 0}

    validation_groups = {"multimer_v3": native_candidates}
    scored: list[dict[str, Any]] = []
    af2_child_runs: dict[str, str] = {}
    for family, family_candidates in validation_groups.items():
        if not family_candidates:
            continue
        family_source_jsonl = validation_dir / f"native_candidates_{family}.jsonl"
        _write_candidate_jsonl(family_source_jsonl, family_candidates)
        update_status(
            run_dir,
            "running",
            current_phase="Common validation",
            current_engine=f"AF2 {family} target-template",
            progress_label=(
                f"Running harmonized {family} complex validation for "
                f"{len(family_candidates)} candidate(s)"
            ),
        )
        af2_run = run_af2_initial_guess_complex_refolding(
            source_run_dir=run_dir,
            candidates_jsonl=family_source_jsonl,
            require_monomer_success=False,
            num_recycles=max(1, int(num_recycles)),
            model_count=2,
            multimer=True,
            binder_multimer=False,
            use_initial_guess=False,
            use_binder_template=False,
            use_interface_template=False,
            gpu_device=gpu_device,
        )
        af2_child_runs[family] = str(af2_run)
        mark_internal_job(
            af2_run,
            parent_run_dir=run_dir,
            parent_task_group=DESIGN_CAMPAIGN_GROUP,
            role="common_af2_target_template_validation",
            engine=f"af2_{family}_target_template",
        )
        af2_candidates = [
            _absolute_candidate_paths(candidate, af2_run)
            for candidate in read_candidates(af2_run)
        ]
        update_status(
            run_dir,
            "running",
            current_phase="Common validation",
            current_engine=f"ipSAE ({family})",
            progress_label=f"Calculating ipSAE for {family} predictions",
        )
        for candidate in _run_ipsae(af2_run, run_dir, af2_candidates):
            metrics = dict(candidate.get("metrics") or {})
            metadata = (
                candidate.get("raw_metadata")
                if isinstance(candidate.get("raw_metadata"), dict)
                else {}
            )
            metrics.update(
                {
                    "design_af2_parameter_family": metadata.get(
                        "design_af2_parameter_family",
                        "none",
                    ),
                    "af2_validation_parameter_family": family,
                    "complex_af2_parameter_family": "multimer_v3",
                    "binder_fold_af2_parameter_family": "ptm",
                    "af2_cross_validation_rule": (
                        "harmonized_multimer_complex_plus_ptm_binder_fold"
                    ),
                }
            )
            candidate["metrics"] = metrics
            scored.append(_add_bindcraft_structure_metrics(candidate))
    confidence_survivors: list[dict[str, Any]] = []
    for candidate in scored:
        metrics = dict(candidate.get("metrics") or {})
        failures = _threshold_failures(metrics, COMMON_AF2_THRESHOLDS)
        ipsae = metrics.get("ipsae")
        if min_ipsae > 0:
            if ipsae in {None, ""}:
                failures.append("ipsae missing")
            elif _number(ipsae) < min_ipsae:
                failures.append(f"ipsae < {min_ipsae:g}")
        metrics.update(
            {
                "common_confidence_pass": not failures,
                "common_confidence_failures": "; ".join(failures),
                "common_validation_profile": "harmonized_multimer_complex_ptm_binder",
            }
        )
        candidate["metrics"] = metrics
        if not failures:
            confidence_survivors.append(candidate)

    update_status(
        run_dir,
        "running",
        current_phase="Common validation",
        current_engine="PyRosetta",
        progress_label=f"Relaxing and scoring {len(confidence_survivors)} confidence survivors",
    )
    rosetta_rows, rosetta_csv, rosetta_rc = _run_common_pyrosetta(
        run_dir,
        confidence_survivors,
        nprocs=pyrosetta_nprocs,
    )
    validated: list[dict[str, Any]] = []
    for candidate in scored:
        metrics = dict(candidate.get("metrics") or {})
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(candidate["candidate_id"]))
        rosetta = rosetta_rows.get(safe_id, {})
        metrics.update({key: _number(value, default=value) for key, value in rosetta.items()})
        confidence_pass = metrics.get("common_confidence_pass") is True
        structure_failures = (
            _threshold_failures(metrics, COMMON_STRUCTURE_THRESHOLDS)
            if confidence_pass
            else ["skipped after common confidence gate"]
        )
        rosetta_failures = (
            _threshold_failures(metrics, COMMON_ROSETTA_THRESHOLDS)
            if confidence_pass
            else ["skipped after common confidence gate"]
        )
        common_pass = confidence_pass and not structure_failures and not rosetta_failures
        metrics.update(
            {
                "common_rosetta_pass": confidence_pass and not rosetta_failures,
                "common_rosetta_failures": "; ".join(rosetta_failures),
                "common_structure_pass": confidence_pass and not structure_failures,
                "common_structure_failures": "; ".join(structure_failures),
                "common_bindcraft_pass": common_pass,
                "common_filter_failures": "; ".join(
                    item
                    for item in [
                        str(metrics.get("common_confidence_failures") or ""),
                        "; ".join(structure_failures),
                        "; ".join(rosetta_failures),
                    ]
                    if item
                ),
                "common_bindcraft_parity_checks": (
                    "AF2-multimer-v3 target-binder complex confidence; "
                    "independent AF2-PTM binder-alone fold prediction; binder fold RMSD; "
                    "binder/interface secondary structure; target-aligned complex pose RMSD; "
                    "PyRosetta"
                ),
            }
        )
        candidate["metrics"] = metrics
        validated.append(candidate)
    _write_candidate_jsonl(validation_dir / "validated_candidates.jsonl", validated)
    write_json(
        validation_dir / "validation_summary.json",
        {
            "profile": "harmonized_multimer_complex_ptm_binder",
            "prediction_input_mode": "target_template_plus_binder_sequence",
            "native_candidate_count": len(native_candidates),
            "candidate_pool_level_counts": pool_level_counts,
            "af2_candidate_count": len(scored),
            "confidence_passing_count": len(confidence_survivors),
            "common_bindcraft_passing_count": sum(
                (candidate.get("metrics") or {}).get("common_bindcraft_pass") is True
                for candidate in validated
            ),
            "af2_thresholds": COMMON_AF2_THRESHOLDS,
            "structure_thresholds": COMMON_STRUCTURE_THRESHOLDS,
            "rosetta_thresholds": COMMON_ROSETTA_THRESHOLDS,
            "min_ipsae": min_ipsae,
            "pyrosetta_return_code": rosetta_rc,
            "pyrosetta_metrics_csv": str(rosetta_csv.relative_to(run_dir)) if rosetta_csv.exists() else "",
            "secondary_structure_scored_count": sum(
                (candidate.get("metrics") or {}).get("average_binder_loop_percent") is not None
                for candidate in validated
            ),
            "hotspot_rmsd_scored_count": sum(
                (candidate.get("metrics") or {}).get("average_hotspot_rmsd") is not None
                for candidate in validated
            ),
            "af2_child_runs": af2_child_runs,
            "validation_family_candidate_counts": {
                family: len(candidates)
                for family, candidates in validation_groups.items()
            },
        },
    )
    return validated, {
        "native_candidate_count": len(native_candidates),
        "candidate_pool_level_counts": pool_level_counts,
        "af2_candidate_count": len(scored),
        "confidence_passing_count": len(confidence_survivors),
        "common_bindcraft_passing_count": sum(
            (candidate.get("metrics") or {}).get("common_bindcraft_pass") is True
            for candidate in validated
        ),
        "secondary_structure_scored_count": sum(
            (candidate.get("metrics") or {}).get("average_binder_loop_percent") is not None
            for candidate in validated
        ),
        "hotspot_rmsd_scored_count": sum(
            (candidate.get("metrics") or {}).get("average_hotspot_rmsd") is not None
            for candidate in validated
        ),
        "af2_child_runs": af2_child_runs,
        "validation_family_candidate_counts": {
            family: len(candidates)
            for family, candidates in validation_groups.items()
        },
        "pyrosetta_return_code": rosetta_rc,
    }


def harmonize_child_candidates(
    child_runs: list[dict[str, Any]],
    *,
    survivors_per_engine: int,
    passing_only: bool,
    keep_best_failed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    harmonized: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    survivor_limit = max(1, int(survivors_per_engine))

    for child in child_runs:
        engine = str(child["engine"])
        child_run = Path(str(child["run_dir"]))
        candidates = [_absolute_candidate_paths(row, child_run) for row in read_candidates(child_run)]
        passing = [row for row in candidates if _native_pass(row) is True]
        if passing_only:
            eligible = passing
            selection_mode = "native_pass"
            if not eligible and keep_best_failed:
                eligible = candidates
                selection_mode = "best_available_fallback"
        else:
            eligible = candidates
            selection_mode = "all"
        selected = sorted(eligible, key=_candidate_sort_key)[:survivor_limit]

        for engine_rank, candidate in enumerate(selected, start=1):
            source_candidate_id = str(candidate.get("candidate_id") or f"candidate-{engine_rank}")
            metrics = dict(candidate.get("metrics") or {})
            metrics.update(
                {
                    "design_campaign_engine_rank": engine_rank,
                    "harmonized_native_pass": _native_pass(candidate),
                    "harmonized_selection_mode": selection_mode,
                }
            )
            raw_metadata = dict(candidate.get("raw_metadata") or {})
            raw_metadata.update(
                {
                    "design_campaign_engine": engine,
                    "source_run_dir": str(child_run),
                    "source_run_id": child_run.name,
                    "source_candidate_id": source_candidate_id,
                }
            )
            candidate.update(
                {
                    "candidate_id": f"{engine}__{source_candidate_id}",
                    "metrics": metrics,
                    "raw_metadata": raw_metadata,
                    "parents": [*list(candidate.get("parents") or []), source_candidate_id],
                }
            )
            harmonized.append(candidate)

        summaries.append(
            {
                "engine": engine,
                "label": ENGINE_LABELS.get(engine, engine),
                "run_dir": str(child_run),
                "run_id": child_run.name,
                "status": child.get("status", ""),
                "candidate_count": len(candidates),
                "native_passing_count": len(passing),
                "selected_count": len(selected),
                "selection_mode": selection_mode,
                "error": child.get("error", ""),
            }
        )

    harmonized.sort(key=lambda row: (ENGINE_ORDER.index(str((row.get("raw_metadata") or {}).get("design_campaign_engine"))), _candidate_sort_key(row)))
    for campaign_rank, candidate in enumerate(harmonized, start=1):
        candidate.setdefault("metrics", {})["design_campaign_rank"] = campaign_rank
    return harmonized, summaries


def harmonize_validated_candidates(
    candidates: list[dict[str, Any]],
    *,
    survivors_per_engine: int,
    keep_best_failed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    harmonized: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    survivor_limit = max(1, int(survivors_per_engine))
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        engine = str((candidate.get("raw_metadata") or {}).get("design_campaign_engine") or "unknown")
        by_engine.setdefault(engine, []).append(candidate)

    def common_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        metrics = candidate.get("metrics") or {}
        return (
            0.0 if metrics.get("common_bindcraft_pass") is True else 1.0,
            -_first_metric(metrics, ("ipsae", "ipSAE", "iptm", "i_ptm"), default=-math.inf),
            _first_metric(metrics, ("ipae", "i_pae")),
            -_first_metric(metrics, ("binder_plddt", "plddt"), default=-math.inf),
            -_first_metric(metrics, ("rosetta_interface_sc",), default=-math.inf),
            _first_metric(metrics, ("rosetta_interface_dG",)),
        )

    for engine in [engine for engine in ENGINE_ORDER if engine in by_engine]:
        engine_candidates = by_engine.get(engine, [])
        passing = [
            candidate
            for candidate in engine_candidates
            if (candidate.get("metrics") or {}).get("common_bindcraft_pass") is True
        ]
        eligible = passing
        selection_mode = "common_bindcraft_pass"
        if not eligible and keep_best_failed:
            eligible = engine_candidates
            selection_mode = "best_common_validation_fallback"
        selected = sorted(eligible, key=common_key)[:survivor_limit]
        for engine_rank, candidate in enumerate(selected, start=1):
            metrics = dict(candidate.get("metrics") or {})
            metrics.update(
                {
                    "design_campaign_engine_rank": engine_rank,
                    "harmonized_selection_mode": selection_mode,
                }
            )
            candidate["metrics"] = metrics
            harmonized.append(candidate)
        summaries.append(
            {
                "engine": engine,
                "label": ENGINE_LABELS.get(engine, engine),
                "candidate_count": len(engine_candidates),
                "common_passing_count": len(passing),
                "selected_count": len(selected),
                "selection_mode": selection_mode,
                "candidate_pool_coverage": CANDIDATE_POOL_COVERAGE.get(engine, "native emitted candidates"),
            }
        )
    harmonized.sort(
        key=lambda row: (
            ENGINE_ORDER.index(str((row.get("raw_metadata") or {}).get("design_campaign_engine"))),
            common_key(row),
        )
    )
    for rank, candidate in enumerate(harmonized, start=1):
        candidate.setdefault("metrics", {})["design_campaign_rank"] = rank
    return harmonized, summaries


def _candidate_structure_path(run_dir: Path, candidate: dict[str, Any]) -> Path | None:
    value = str(candidate.get("complex_pdb") or candidate.get("binder_pdb") or "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _binder_interface_residues(
    structure: Path,
    binder_chains: list[str],
    target_chains: list[str],
    *,
    cutoff: float = 5.0,
) -> list[str]:
    binder_set = set(binder_chains)
    target_set = set(target_chains)
    binder_atoms: list[tuple[str, tuple[float, float, float]]] = []
    target_atoms: list[tuple[float, float, float]] = []
    for line in structure.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper()
        if element == "H" or (not element and atom_name.startswith("H")):
            continue
        chain = line[21].strip() or "_"
        try:
            coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if chain in binder_set:
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            if residue_number:
                binder_atoms.append((f"{chain}{residue_number}{insertion_code}", coords))
        elif chain in target_set:
            target_atoms.append(coords)
    if not binder_atoms or not target_atoms:
        return []

    cell_size = max(0.1, float(cutoff))
    target_grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for coords in target_atoms:
        cell = tuple(math.floor(value / cell_size) for value in coords)
        target_grid.setdefault(cell, []).append(coords)
    cutoff_sq = cutoff * cutoff
    contacts: set[str] = set()
    for residue, coords in binder_atoms:
        cell = tuple(math.floor(value / cell_size) for value in coords)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for target in target_grid.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), []
                    ):
                        if sum((coords[index] - target[index]) ** 2 for index in range(3)) <= cutoff_sq:
                            contacts.add(residue)
                            break
                    if residue in contacts:
                        break
                if residue in contacts:
                    break
            if residue in contacts:
                break
    return sorted(
        contacts,
        key=lambda tag: (
            tag[0],
            int("".join(ch for ch in tag[1:] if ch.isdigit()) or 0),
            tag,
        ),
    )


def run_campaign_sequence_refinement(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    gpu_device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(config.get("mode") or "none")
    if mode == "none":
        return candidates, {"mode": "none", "status": "skipped", "candidate_count": len(candidates)}
    if mode not in {
        "redesign_non_interface",
        "redesign_all",
        "redesign_selected_unlocked",
        "redesign_selected_locked",
    }:
        raise ValueError(f"Unsupported sequence refinement mode: {mode}")

    refinement_dir = run_dir / "artifacts" / "design_campaign" / "sequence_refinement"
    fixed_residues: dict[str, list[str]] = {}
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        structure = _candidate_structure_path(run_dir, candidate)
        if not candidate_id or structure is None or not structure.exists() or structure.suffix.lower() != ".pdb":
            skipped.append({"candidate_id": candidate_id, "reason": "missing PDB complex"})
            continue
        structure_chains = {
            line[21].strip() or "_"
            for line in structure.read_text(errors="ignore").splitlines()
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21
        }
        declared_targets = {
            str(chain) for chain in candidate.get("target_chains") or [] if str(chain) in structure_chains
        }
        binder_chains = [
            str(chain)
            for chain in candidate.get("binder_chains") or []
            if str(chain) in structure_chains and str(chain) not in declared_targets
        ]
        if not binder_chains:
            binder_chains = sorted(structure_chains - declared_targets)[-1:]
        target_chains = [
            str(chain)
            for chain in candidate.get("target_chains") or []
            if str(chain) in structure_chains and str(chain) not in set(binder_chains)
        ]
        if not target_chains:
            target_chains = sorted(structure_chains - set(binder_chains))
        if mode == "redesign_non_interface":
            contacts = _binder_interface_residues(structure, binder_chains, target_chains)
            if not contacts:
                skipped.append({"candidate_id": candidate_id, "reason": "no binder interface residues found"})
                continue
            fixed_residues[candidate_id] = contacts
        elif mode == "redesign_selected_unlocked":
            unlocked = set(
                _parse_residue_tags(
                    config.get("unlocked_residues"),
                    default_chain=binder_chains[0] if binder_chains else "",
                )
            )
            if not unlocked:
                skipped.append({"candidate_id": candidate_id, "reason": "no residues selected for redesign"})
                continue
            binder_residues = _pdb_residue_tags(structure, binder_chains)
            fixed_residues[candidate_id] = [tag for tag in binder_residues if tag not in unlocked]
        elif mode == "redesign_selected_locked":
            locked = _parse_residue_tags(
                config.get("locked_residues"),
                default_chain=binder_chains[0] if binder_chains else "",
            )
            if not locked:
                skipped.append({"candidate_id": candidate_id, "reason": "no residues selected to lock"})
                continue
            fixed_residues[candidate_id] = locked
        eligible.append(candidate)

    source_jsonl = refinement_dir / "source_candidates.jsonl"
    _write_candidate_jsonl(source_jsonl, eligible)
    write_json(refinement_dir / "skipped_candidates.json", {"candidates": skipped})
    if not eligible:
        return candidates, {
            "mode": mode,
            "status": "failed",
            "error": "No candidates were eligible for sequence refinement.",
            "input_candidate_count": len(candidates),
            "skipped_candidate_count": len(skipped),
        }

    model_type = REFINEMENT_MODEL_TYPES.get(
        str(config.get("engine") or "protein_mpnn"), "protein_mpnn"
    )
    update_status(
        run_dir,
        "running",
        current_phase="Sequence refinement",
        current_engine=model_type,
        progress_label=f"Refining {len(eligible)} harmonized candidates with {model_type}",
    )
    child_run = run_ligandmpnn_sequence_design(
        source_run_dir=run_dir,
        candidates_jsonl=source_jsonl,
        model_type=model_type,
        num_seq_per_target=max(1, int(config.get("sequences_per_structure", 2))),
        sampling_temp=float(config.get("sampling_temp", 0.1)),
        omit_aas=str(config.get("omit_aas") or "CX"),
        seed=int(config.get("seed", 0)),
        accepted_stages=sorted({str(candidate.get("stage") or "") for candidate in eligible}),
        fixed_residues_by_candidate=fixed_residues,
        gpu_device=gpu_device,
    )
    mark_internal_job(
        child_run,
        parent_run_dir=run_dir,
        parent_task_group=DESIGN_CAMPAIGN_GROUP,
        role="sequence_refinement",
        engine=model_type,
    )
    child_result = read_json(child_run / "result.json")
    parent_by_id = {str(candidate["candidate_id"]): candidate for candidate in eligible}
    refined: list[dict[str, Any]] = []
    for row in read_candidates(child_run):
        row = _absolute_candidate_paths(row, child_run)
        parent_id = str((row.get("parents") or [""])[0])
        parent = parent_by_id.get(parent_id)
        if parent is None:
            continue
        sequence_metrics = dict(row.get("metrics") or {})
        metrics = dict(parent.get("metrics") or {})
        metrics.update({f"refinement_{key}": value for key, value in sequence_metrics.items()})
        metrics.update(
            {
                "sequence_refinement_mode": mode,
                "sequence_refinement_engine": model_type,
                "sequence_refinement_fixed_residue_count": len(fixed_residues.get(parent_id, [])),
                "post_refinement_validation_required": True,
            }
        )
        metadata = dict(parent.get("raw_metadata") or {})
        metadata.update(
            {
                "sequence_refinement_run_dir": str(child_run),
                "sequence_refinement_run_id": child_run.name,
                "sequence_refinement_parent_id": parent_id,
                "sequence_refinement_fixed_residues": fixed_residues.get(parent_id, []),
                "pre_refinement_candidate": parent,
            }
        )
        row.update(
            {
                "metrics": metrics,
                "raw_metadata": metadata,
                "parents": [parent_id, *list(parent.get("parents") or [])],
            }
        )
        refined.append(row)

    success = child_result.get("success") is True and bool(refined)
    summary = {
        "mode": mode,
        "status": "completed" if success else "failed",
        "engine": model_type,
        "input_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "skipped_candidate_count": len(skipped),
        "refined_candidate_count": len(refined),
        "sequences_per_structure": max(1, int(config.get("sequences_per_structure", 2))),
        "fixed_interface_candidate_count": len(fixed_residues),
        "child_run": str(child_run),
        "error": "" if success else str((child_result.get("metrics") or {}).get("error") or "Refinement failed"),
    }
    write_json(refinement_dir / "refinement_summary.json", summary)
    return (refined if success else candidates), summary


def run_campaign_evaluation(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    gpu_device: str,
) -> dict[str, Any]:
    mode = str(config.get("mode") or "none")
    if mode == "none":
        return {"mode": "none", "status": "skipped", "candidate_count": len(candidates)}
    if mode not in {"metrics_only", "refold_and_metrics"}:
        raise ValueError(f"Unsupported campaign evaluation mode: {mode}")

    run_ipsae = bool(config.get("ipsae", True))
    run_rosetta = bool(config.get("rosetta", False))
    run_pymol = bool(config.get("pymol", False))
    if mode == "metrics_only" and not any((run_ipsae, run_rosetta, run_pymol)):
        return {
            "mode": mode,
            "status": "skipped",
            "candidate_count": len(candidates),
            "reason": "No evaluation metrics were selected.",
        }
    if not candidates:
        return {"mode": mode, "status": "failed", "candidate_count": 0, "error": "No candidates to evaluate."}

    from mn_protein_design.workflows import benchmark as benchmark_workflow

    evaluation_dir = run_dir / "artifacts" / "design_campaign" / "evaluation"
    source_jsonl = evaluation_dir / "source_candidates.jsonl"
    _write_candidate_jsonl(source_jsonl, candidates)
    refolder = str(config.get("refolder") or "AF3")
    run_refolding = mode == "refold_and_metrics"
    refolder_flags = {
        "run_alphafast_af3": run_refolding and refolder == "AF3",
        "run_esmfold2": run_refolding and refolder == "ESMFold2",
        "run_boltz2_initial_guess": run_refolding and refolder == "Boltz-2",
        "run_rf3": run_refolding and refolder == "RF3",
    }
    update_status(
        run_dir,
        "running",
        current_phase="Evaluation",
        current_engine=refolder if run_refolding else "Metrics",
        progress_label=(
            f"Refolding and evaluating {len(candidates)} candidates with {refolder}"
            if run_refolding
            else f"Evaluating {len(candidates)} existing candidate structures"
        ),
    )
    child_run = benchmark_workflow.run_de_novo_binder_scoring_dataset(
        source_run_dir=run_dir,
        candidates_jsonl=source_jsonl,
        selected_candidate_ids=[str(candidate.get("candidate_id") or "") for candidate in candidates],
        mode="seq_only_csv",
        generate_inputs=True,
        models=[],
        run_common_interface_metrics=run_ipsae,
        run_pyrosetta_input_metrics=run_rosetta,
        run_predicted_rosetta_metrics=run_rosetta and run_refolding,
        run_pymol_metrics=run_pymol,
        pyrosetta_nprocs=max(1, int(config.get("pyrosetta_nprocs", 4))),
        num_loops=max(1, int(config.get("num_recycles", 10))),
        num_sampling_steps=max(1, int(config.get("num_sampling_steps", 68))),
        seed=max(0, int(config.get("seed", 0))),
        device="cuda",
        esmfold2_modes=["sequence"],
        esmfold2_use_target_msa=bool(config.get("use_target_msa", False)),
        boltz2_use_target_template=True,
        boltz2_use_target_msa=bool(config.get("use_target_msa", True)),
        boltz2_recycling_steps=max(1, int(config.get("num_recycles", 10))),
        boltz2_sampling_steps=max(1, int(config.get("num_sampling_steps", 200))),
        boltz2_diffusion_samples=max(1, int(config.get("num_samples", 3))),
        rf3_use_target_msa=bool(config.get("use_target_msa", True)),
        rf3_recycles=max(1, int(config.get("num_recycles", 10))),
        rf3_num_steps=max(1, int(config.get("num_sampling_steps", 50))),
        rf3_diffusion_batch_size=max(1, int(config.get("num_samples", 5))),
        rf3_seed=max(0, int(config.get("seed", 0))),
        alphafast_num_recycles=max(1, int(config.get("num_recycles", 10))),
        alphafast_gpu_device=gpu_device,
        gpu_device=gpu_device,
        max_records=0,
        job_type="design_campaign_evaluation",
        tool_name="design_campaign_evaluation",
        **refolder_flags,
    )
    mark_internal_job(
        child_run,
        parent_run_dir=run_dir,
        parent_task_group=DESIGN_CAMPAIGN_GROUP,
        role="campaign_evaluation",
        engine=refolder if run_refolding else "metrics_only",
    )
    result = read_json(child_run / "result.json")
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    copied_outputs: dict[str, str] = {}
    for name in (
        "merged_benchmark_metrics",
        "common_interface_metrics",
        "input_rosetta_metrics",
        "predicted_rosetta_metrics",
    ):
        value = outputs.get(name)
        if not isinstance(value, str) or not value:
            continue
        source = child_run / value
        if source.exists() and source.is_file():
            destination = evaluation_dir / source.name
            shutil.copy2(source, destination)
            copied_outputs[name] = str(destination.relative_to(run_dir))
    summary = {
        "mode": mode,
        "status": "completed" if result.get("success") is True else "failed",
        "candidate_count": len(candidates),
        "refolder": refolder if run_refolding else "",
        "ipsae": run_ipsae,
        "rosetta": run_rosetta,
        "pymol": run_pymol,
        "child_run": str(child_run),
        "child_run_id": child_run.name,
        "copied_outputs": copied_outputs,
        "engine_artifacts": metrics.get("engine_artifacts", {}),
        "error": "" if result.get("success") is True else str(metrics.get("error") or "Evaluation failed"),
    }
    write_json(evaluation_dir / "evaluation_summary.json", summary)
    return summary


def _run_engine(
    engine: str,
    *,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str,
    campaign_name: str,
    design_attempts: int,
    sequences_per_backbone: int,
    random_seed: int,
    gpu_device: str,
    config: dict[str, Any],
) -> Path:
    runners: dict[str, Callable[..., Path]] = {
        "bindcraft": run_bindcraft,
        "rfdiffusion3_foundry": run_rfdiffusion3_foundry,
        "boltzgen": run_boltzgen,
        "rfdiffusion_classic": run_rfdiffusion_classic,
        "pxdesign": run_pxdesign,
        "genie3": run_genie3,
        "esmfold2_binder_design": run_esmfold2_native_binder_design,
        "protpardelle_1c": run_protpardelle_1c,
        "proteina_complexa": run_proteina_complexa,
    }
    if engine not in runners:
        raise ValueError(f"Unsupported design campaign engine: {engine}")

    shared: dict[str, Any] = {
        "target_pdb": target_pdb,
        "hotspots": hotspots,
        "campaign_name": campaign_name,
        "gpu_device": gpu_device,
    }
    if engine == "esmfold2_binder_design":
        lengths = [int(part) for part in binder_length.replace("-", " ").split() if part.isdigit()]
        if not lengths:
            raise ValueError("ESMFold2 binder design requires an integer binder length.")
        return runners[engine](
            **shared,
            target_chain=str(config.get("target_chain") or target_chains[0]),
            binder_length=int(config.get("binder_length") or lengths[0]),
            num_designs=design_attempts,
            optimization_steps=int(config.get("optimization_steps", 150)),
            learning_rate=float(config.get("learning_rate", 0.1)),
            seed=random_seed,
            compile_model=bool(config.get("compile_model", False)),
            checkpoint_lm=bool(config.get("checkpoint_lm", False)),
        )

    shared.update({"target_chains": target_chains, "binder_length": binder_length})
    if engine == "rfdiffusion_classic":
        contig = str(config.get("contig") or default_target_contig(target_pdb, target_chains, binder_length))
        return runners[engine](
            **shared,
            contig=contig,
            num_designs=design_attempts,
            timesteps=int(config.get("timesteps", 50)),
            model_weights=str(config.get("model_weights", "Complex_base")),
            rfdiffusion_guidance_preset=str(config.get("guidance_preset", "none")),
            mpnn_num_sequences=sequences_per_backbone,
            sequence_design_method=str(config.get("sequence_design_method", "ligandmpnn")),
            execution_backend=str(config.get("execution_backend", "docker")),
            run_vanilla_pipeline=True,
            analysis_keep_top_n=max(1, int(config.get("analysis_keep_top_n", design_attempts))),
        )
    if engine == "bindcraft":
        return runners[engine](
            **shared,
            number_of_final_designs=min(int(config.get("number_of_final_designs", 1)), design_attempts),
            num_seqs_override=sequences_per_backbone,
            max_mpnn_sequences_override=sequences_per_backbone,
            time_limit_seconds=int(config["time_limit_seconds"]) if config.get("time_limit_seconds") else None,
            max_trajectories=design_attempts,
            filter_settings=str(config.get("filter_settings", "default_filters.json")),
        )
    if engine == "rfdiffusion3_foundry":
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            timesteps=int(config.get("timesteps", 50)),
            run_vanilla_pipeline=True,
            mpnn_sequences_per_backbone=sequences_per_backbone,
            mpnn_model_type=str(config.get("mpnn_model_type", "protein_mpnn")),
            mpnn_checkpoint_path=str(config.get("mpnn_checkpoint_path", "/weights/proteinmpnn_v_48_020.pt")),
            prepare_target_msa=bool(config.get("prepare_target_msa", True)),
        )
    if engine == "boltzgen":
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            budget=min(int(config.get("budget", 1)), design_attempts),
            sampling_steps=int(config.get("sampling_steps", 20)),
            run_vanilla_pipeline=True,
        )
    if engine == "pxdesign":
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            n_steps=int(config.get("n_steps", 400)),
            dtype=str(config.get("dtype", "bf16")),
            preset=str(config.get("preset", "preview")),
            run_mode="pipeline",
            n_max_runs=int(config.get("n_max_runs", 1)),
            use_fast_ln=bool(config.get("use_fast_ln", True)),
            use_deepspeed_evo_attention=bool(config.get("use_deepspeed_evo_attention", False)),
            prepare_target_msa=bool(config.get("prepare_target_msa", True)),
        )
    if engine == "genie3":
        folding_model = str(config.get("folding_model_name", "colabfold"))
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            seed=random_seed,
            direction_scale=float(config.get("direction_scale", 0.0)),
            cond_strategy=str(config.get("cond_strategy", "extended")),
            inverse_folding_num_seq=sequences_per_backbone,
            folding_model_name=folding_model,
            folding_mode=str(config.get("folding_mode", "msa" if folding_model == "boltz2" else "template")),
            folding_num_models=int(config.get("folding_num_models", 5)),
            folding_num_recycles=int(config.get("folding_num_recycles", 20)),
            compile_generation=bool(config.get("compile_generation", False)),
            run_mode="full_vanilla_pipeline",
            enable_beam_search=bool(config.get("enable_beam_search", False)),
            beam_width=int(config.get("beam_width", 4)),
        )
    if engine == "protpardelle_1c":
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            num_mpnn_seqs=sequences_per_backbone,
            model_name=str(config.get("model_name", "cc83")),
            model_epoch=str(config.get("model_epoch", "2616")),
            sampling_config=str(config.get("sampling_config", "sampling_sidechain_conditional")),
            step_scale=float(config.get("step_scale", 1.2)),
            schurn=int(config.get("schurn", 200)),
            crop_cond_start=float(config.get("crop_cond_start", 0.0)),
            batch_size=int(config.get("batch_size", 1)),
            seed=random_seed,
        )
    return runners[engine](
        **shared,
        num_designs=design_attempts,
        n_steps=int(config.get("n_steps", 400)),
        replicas=int(config.get("replicas", 2)),
        seed=random_seed,
        batch_size=int(config.get("batch_size", 1)),
    )


def _write_summary_csv(run_dir: Path, candidates: list[dict[str, Any]]) -> Path:
    output = run_dir / "artifacts" / "design_campaign" / "harmonized_candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "design_campaign_rank",
        "engine",
        "engine_rank",
        "candidate_id",
        "stage",
        "native_pass",
        "binder_sequence",
        "complex_pdb",
        "binder_pdb",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            metrics = candidate.get("metrics") or {}
            metadata = candidate.get("raw_metadata") or {}
            writer.writerow(
                {
                    "design_campaign_rank": metrics.get("design_campaign_rank"),
                    "engine": metadata.get("design_campaign_engine"),
                    "engine_rank": metrics.get("design_campaign_engine_rank"),
                    "candidate_id": candidate.get("candidate_id"),
                    "stage": candidate.get("stage"),
                    "native_pass": metrics.get("harmonized_native_pass"),
                    "binder_sequence": candidate.get("binder_sequence"),
                    "complex_pdb": candidate.get("complex_pdb"),
                    "binder_pdb": candidate.get("binder_pdb"),
                }
            )
    return output


def run_design_campaign(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    payload = read_json(run_dir / "input.json")
    inputs = payload.get("inputs") or {}
    params = payload.get("params") or {}
    target_pdb = Path(str(inputs.get("target_pdb") or "")).expanduser()
    target_chains = [str(chain) for chain in inputs.get("target_chains") or []]
    engines = [
        engine
        for engine in params.get("engines") or []
        if engine in ENGINE_ORDER and engine != "template_redesign"
    ]
    engine_configs = params.get("engine_configs") if isinstance(params.get("engine_configs"), dict) else {}
    campaign_name = str(params.get("campaign_name") or "").strip()
    continue_after_failure = bool(params.get("continue_after_failure", True))
    design_attempts = max(1, int(params.get("design_attempts", 1)))
    sequences_per_backbone = max(1, int(params.get("sequences_per_backbone", 1)))
    random_seed = max(0, int(params.get("random_seed", 0)))
    common_config = params.get("common_validation") if isinstance(params.get("common_validation"), dict) else {}
    template_config = (
        params.get("template_redesign")
        if isinstance(params.get("template_redesign"), dict)
        else {"enabled": False}
    )
    template_enabled = bool(template_config.get("enabled"))
    refinement_config = (
        params.get("sequence_refinement")
        if isinstance(params.get("sequence_refinement"), dict)
        else {"mode": "none"}
    )
    evaluation_config = (
        params.get("evaluation")
        if isinstance(params.get("evaluation"), dict)
        else {"mode": "none"}
    )
    if template_enabled:
        template_mode_defaults = {
            "interface": "redesign_non_interface",
            "all": "redesign_all",
            "selected_unlocked": "redesign_selected_unlocked",
            "selected_locked": "redesign_selected_locked",
        }
        if str(refinement_config.get("mode") or "none") == "none":
            refinement_config["mode"] = template_mode_defaults.get(
                str(template_config.get("mode") or "interface"),
                "redesign_non_interface",
            )
        refinement_config.setdefault("locked_residues", template_config.get("locked_residues", ""))
        refinement_config.setdefault("unlocked_residues", template_config.get("unlocked_residues", ""))

    if not target_pdb.exists():
        raise FileNotFoundError(f"Target PDB does not exist: {target_pdb}")
    if not target_chains:
        raise ValueError("At least one target chain is required.")
    if not engines and not template_enabled:
        raise ValueError("At least one vanilla workflow or template redesign source is required.")

    update_status(
        run_dir,
        "running",
        campaign_name=campaign_name,
        current_phase="Vanilla workflows",
        current_engine="",
        completed_engines=0,
        total_engines=len(engines) + int(template_enabled),
    )
    child_runs: list[dict[str, Any]] = []
    for index, engine in enumerate(engines, start=1):
        label = ENGINE_LABELS[engine]
        update_status(
            run_dir,
            "running",
            current_phase="Vanilla workflows",
            current_engine=label,
            progress_label=f"Running {label} ({index}/{len(engines)})",
            completed_engines=index - 1,
        )
        try:
            engine_config = dict(engine_configs.get(engine) or {})
            if engine == "bindcraft" and bool(common_config.get("enabled", True)):
                engine_config.update(
                    {
                        "number_of_final_designs": design_attempts,
                        "filter_settings": "no_filters.json",
                    }
                )
            if engine == "boltzgen" and bool(common_config.get("enabled", True)):
                engine_config["budget"] = design_attempts
            child_run = _run_engine(
                engine,
                target_pdb=target_pdb,
                target_chains=target_chains,
                binder_length=str(params.get("binder_length") or "60"),
                hotspots=str(params.get("hotspots") or ""),
                campaign_name=campaign_name,
                design_attempts=design_attempts,
                sequences_per_backbone=sequences_per_backbone,
                random_seed=random_seed,
                gpu_device=str(params.get("gpu_device") or "0"),
                config=engine_config,
            )
            child_result = read_json(child_run / "result.json")
            child_success = child_result.get("success") is True
            child_runs.append(
                {
                    "engine": engine,
                    "run_dir": str(child_run),
                    "run_id": child_run.name,
                    "status": "completed" if child_success else "failed",
                    "error": "" if child_success else str((child_result.get("metrics") or {}).get("error") or "Workflow failed"),
                }
            )
            mark_internal_job(
                child_run,
                parent_run_dir=run_dir,
                parent_task_group=DESIGN_CAMPAIGN_GROUP,
                role="vanilla_design_workflow",
                engine=engine,
            )
            child_metadata = read_json(child_run / "metadata.json")
            child_metadata["design_campaign_engine"] = engine
            write_json(child_run / "metadata.json", child_metadata)
            if not child_success and not continue_after_failure:
                break
        except Exception as exc:
            child_runs.append(
                {
                    "engine": engine,
                    "run_dir": "",
                    "run_id": "",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not continue_after_failure:
                break
        update_status(run_dir, "running", completed_engines=index)
    if template_enabled:
        update_status(
            run_dir,
            "running",
            current_phase="Template redesign",
            current_engine=ENGINE_LABELS["template_redesign"],
            progress_label="Registering uploaded target-binder complex",
        )
        try:
            template_child = _create_template_redesign_source(
                run_dir,
                target_pdb=target_pdb,
                template_config=template_config,
            )
            if template_child is not None:
                child_runs.append(template_child)
                update_status(
                    run_dir,
                    "running",
                    completed_engines=len(engines) + 1,
                )
        except Exception as exc:
            child_runs.append(
                {
                    "engine": "template_redesign",
                    "run_dir": "",
                    "run_id": "",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not continue_after_failure:
                raise

    readable_children = [row for row in child_runs if row.get("run_dir")]
    update_status(
        run_dir,
        "running",
        current_phase="Harmonize",
        current_engine="",
        progress_label="Selecting pass-first survivors",
    )
    common_summary: dict[str, Any] = {}
    if bool(common_config.get("enabled", True)):
        validated, common_summary = run_common_bindcraft_validation(
            run_dir,
            readable_children,
            gpu_device=str(params.get("gpu_device") or "0"),
            num_recycles=int(common_config.get("num_recycles", 3)),
            min_ipsae=float(common_config.get("min_ipsae", 0.0)),
            pyrosetta_nprocs=int(common_config.get("pyrosetta_nprocs", 4)),
        )
        harmonized, summaries = harmonize_validated_candidates(
            validated,
            survivors_per_engine=int(params.get("survivors_per_engine", 2)),
            keep_best_failed=bool(params.get("keep_best_failed", True)),
        )
    else:
        harmonized, summaries = harmonize_child_candidates(
            readable_children,
            survivors_per_engine=int(params.get("survivors_per_engine", 2)),
            passing_only=bool(params.get("passing_only", True)),
            keep_best_failed=bool(params.get("keep_best_failed", True)),
        )
    summary_by_engine = {row["engine"]: row for row in summaries}
    for child in child_runs:
        if child["engine"] not in summary_by_engine:
            summaries.append(
                {
                    "engine": child["engine"],
                    "label": ENGINE_LABELS.get(child["engine"], child["engine"]),
                    "run_dir": child.get("run_dir", ""),
                    "run_id": child.get("run_id", ""),
                    "status": child.get("status", "failed"),
                    "candidate_count": 0,
                    "native_passing_count": 0,
                    "selected_count": 0,
                    "selection_mode": "not_available",
                    "error": child.get("error", ""),
                }
            )
    refined_candidates, refinement_summary = run_campaign_sequence_refinement(
        run_dir,
        harmonized,
        config={
            **refinement_config,
            "seed": random_seed,
        },
        gpu_device=str(params.get("gpu_device") or "0"),
    )
    for final_rank, candidate in enumerate(refined_candidates, start=1):
        metrics = candidate.setdefault("metrics", {})
        if "design_campaign_rank" in metrics and "design_campaign_parent_rank" not in metrics:
            metrics["design_campaign_parent_rank"] = metrics["design_campaign_rank"]
        metrics["design_campaign_rank"] = final_rank

    try:
        evaluation_summary = run_campaign_evaluation(
            run_dir,
            refined_candidates,
            config={
                **evaluation_config,
                "seed": random_seed,
            },
            gpu_device=str(params.get("gpu_device") or "0"),
        )
    except Exception as exc:
        evaluation_summary = {
            "mode": str(evaluation_config.get("mode") or "none"),
            "status": "failed",
            "candidate_count": len(refined_candidates),
            "error": f"{type(exc).__name__}: {exc}",
        }
        evaluation_dir = run_dir / "artifacts" / "design_campaign" / "evaluation"
        write_json(evaluation_dir / "evaluation_summary.json", evaluation_summary)
    for candidate in refined_candidates:
        metrics = candidate.setdefault("metrics", {})
        metrics.update(
            {
                "campaign_evaluation_mode": evaluation_summary.get("mode"),
                "campaign_evaluation_status": evaluation_summary.get("status"),
                "campaign_evaluation_refolder": evaluation_summary.get("refolder", ""),
            }
        )
        metadata = candidate.setdefault("raw_metadata", {})
        metadata["campaign_evaluation_child_run"] = evaluation_summary.get("child_run", "")

    normalized = write_candidates(run_dir, "design_campaign", refined_candidates)
    summary_path = run_dir / "artifacts" / "design_campaign" / "workflow_summary.json"
    write_json(
        summary_path,
        {
            "campaign_name": campaign_name,
            "target_pdb": str(target_pdb),
            "target_chains": target_chains,
            "engines": engines,
            "template_redesign": template_config,
            "workflows": summaries,
            "candidate_count": len(normalized),
            "harmonized_candidate_count": len(harmonized),
            "common_validation": common_summary,
            "sequence_refinement": refinement_summary,
            "evaluation": evaluation_summary,
        },
    )
    csv_path = _write_summary_csv(run_dir, normalized)
    finish_job(
        run_dir,
        bool(normalized),
        {
            "outputs": {
                "candidates": normalized,
                "child_runs": child_runs,
                "workflow_summary": str(summary_path.relative_to(run_dir)),
                "harmonized_candidates_csv": str(csv_path.relative_to(run_dir)),
            },
            "metrics": {
                "engine_count": len(engines),
                "completed_engine_count": sum(row.get("status") == "completed" for row in child_runs),
                "failed_engine_count": sum(row.get("status") == "failed" for row in child_runs),
                "candidate_count": len(normalized),
                "harmonized_candidate_count": len(harmonized),
                "sequence_refinement_status": refinement_summary.get("status"),
                "refined_candidate_count": refinement_summary.get("refined_candidate_count", 0),
                "evaluation_status": evaluation_summary.get("status"),
                "evaluation_mode": evaluation_summary.get("mode"),
                **common_summary,
            },
            "downstream_artifacts": {
                "target_pdb": str(target_pdb),
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return run_dir


def create_design_campaign(
    *,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str,
    campaign_name: str,
    design_attempts: int,
    sequences_per_backbone: int,
    random_seed: int,
    engines: list[str],
    engine_configs: dict[str, dict[str, Any]],
    survivors_per_engine: int = 2,
    passing_only: bool = True,
    keep_best_failed: bool = True,
    continue_after_failure: bool = True,
    common_validation: dict[str, Any] | None = None,
    sequence_refinement: dict[str, Any] | None = None,
    template_redesign: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    gpu_device: str = "0",
) -> Path:
    selected_engines = [engine for engine in ENGINE_ORDER if engine in engines and engine != "template_redesign"]
    params = {
        "campaign_name": campaign_name.strip(),
        "binder_length": binder_length.strip(),
        "hotspots": hotspots.strip(),
        "design_attempts": max(1, int(design_attempts)),
        "sequences_per_backbone": max(1, int(sequences_per_backbone)),
        "random_seed": max(0, int(random_seed)),
        "engines": selected_engines,
        "engine_configs": engine_configs,
        "survivors_per_engine": int(survivors_per_engine),
        "passing_only": bool(passing_only),
        "keep_best_failed": bool(keep_best_failed),
        "continue_after_failure": bool(continue_after_failure),
        "common_validation": dict(
            common_validation
            or {
                "enabled": True,
                "profile": "bindcraft_default_target_template",
                "num_recycles": 3,
                "min_ipsae": 0.0,
                "pyrosetta_nprocs": 4,
            }
        ),
        "sequence_refinement": dict(sequence_refinement or {"mode": "none"}),
        "template_redesign": dict(template_redesign or {"enabled": False}),
        "evaluation": dict(evaluation or {"mode": "none"}),
        "gpu_device": str(gpu_device),
    }
    job = create_job(
        DESIGN_CAMPAIGN_GROUP,
        "multi_engine_design_campaign",
        "design_campaign",
        {"target_pdb": str(target_pdb), "target_chains": target_chains},
        params,
    )
    write_json(
        job.run_dir / "worker_request.json",
        {"kind": "design_campaign", "kwargs": {}, "path_kwargs": []},
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "local_worker",
            "command": [
                "python",
                "-m",
                "mn_protein_design.core.local_worker",
                "--run-dir",
                str(job.run_dir),
            ],
        },
    )
    if campaign_name.strip():
        update_status(job.run_dir, "queued", campaign_name=campaign_name.strip())
    spawn_worker_for_run(job.run_dir)
    return job.run_dir


def create_design_campaign_collection(
    *,
    name: str,
    source_run_ids: list[str],
    target_key: str,
) -> Path:
    clean_name = str(name or "Design campaign collection").strip() or "Design campaign collection"
    selected_run_ids = [str(run_id).strip() for run_id in source_run_ids if str(run_id).strip()]
    job = create_job(
        DESIGN_CAMPAIGN_GROUP,
        "design_campaign_collection",
        "design_campaign_collection_merge",
        {"source_runs": selected_run_ids},
        {"collection_name": clean_name, "target_key": str(target_key or "")},
    )
    update_status(job.run_dir, "running", campaign_name=clean_name)

    collection_dir = job.run_dir / "artifacts" / "design_campaign_collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    workflows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    metrics_total = {
        "candidate_count": 0,
        "harmonized_candidate_count": 0,
        "native_candidate_count": 0,
        "common_bindcraft_passing_count": 0,
        "completed_engine_count": 0,
        "failed_engine_count": 0,
    }
    seen_candidate_ids: set[str] = set()

    for order, run_id in enumerate(selected_run_ids, start=1):
        source_run_dir = runs_root() / DESIGN_CAMPAIGN_GROUP / run_id
        input_payload = read_json(source_run_dir / "input.json")
        metadata = read_json(source_run_dir / "metadata.json")
        result = read_json(source_run_dir / "result.json")
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        source_name = str(params.get("campaign_name") or metadata.get("campaign_name") or run_id)
        source_label = f"{metadata.get('job_code') or run_id[:5]} | {source_name}"
        source_candidates = read_candidates(source_run_dir)
        source_summary = read_json(source_run_dir / "artifacts" / "design_campaign" / "workflow_summary.json")
        source_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        for key in metrics_total:
            try:
                metrics_total[key] += int(source_metrics.get(key) or 0)
            except (TypeError, ValueError):
                pass
        for candidate in source_candidates:
            merged = dict(candidate)
            candidate_id = str(merged.get("candidate_id") or "")
            if candidate_id in seen_candidate_ids:
                candidate_id = f"{candidate_id}__{metadata.get('job_code') or run_id[:5]}"
                merged["candidate_id"] = candidate_id
            seen_candidate_ids.add(candidate_id)
            raw_metadata = (
                dict(merged.get("raw_metadata"))
                if isinstance(merged.get("raw_metadata"), dict)
                else {}
            )
            candidate_metrics = (
                dict(merged.get("metrics"))
                if isinstance(merged.get("metrics"), dict)
                else {}
            )
            raw_metadata.update(
                {
                    "design_campaign_collection": clean_name,
                    "design_campaign_source_run_id": run_id,
                    "design_campaign_source_job_code": str(metadata.get("job_code") or ""),
                    "design_campaign_source_campaign": source_label,
                    "design_campaign_collection_order": order,
                }
            )
            candidate_metrics["source_campaign"] = source_label
            merged["raw_metadata"] = raw_metadata
            merged["metrics"] = candidate_metrics
            candidates.append(merged)
        for workflow in source_summary.get("workflows") if isinstance(source_summary.get("workflows"), list) else []:
            workflow_row = dict(workflow)
            workflow_row["campaign"] = source_label
            workflows.append(workflow_row)
        source_rows.append(
            {
                "order": order,
                "run_id": run_id,
                "job_code": metadata.get("job_code"),
                "campaign": source_name,
                "status": metadata.get("status"),
                "candidates": len(source_candidates),
                "completed_engines": source_metrics.get("completed_engine_count"),
                "failed_engines": source_metrics.get("failed_engine_count"),
            }
        )

    if not candidates:
        finish_job(
            job.run_dir,
            False,
            {"metrics": {"error": "No source candidates were available."}, "outputs": {}},
        )
        raise ValueError("No source candidates were available.")

    normalized = write_candidates(job.run_dir, "design_campaign_collection", candidates)
    sources_path = collection_dir / "design_campaign_collection_sources.csv"
    with sources_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "order",
                "run_id",
                "job_code",
                "campaign",
                "status",
                "candidates",
                "completed_engines",
                "failed_engines",
            ],
        )
        writer.writeheader()
        writer.writerows(source_rows)
    collection_path = collection_dir / "collection.json"
    write_json(
        collection_path,
        {
            "name": clean_name,
            "target_key": target_key,
            "source_run_ids": selected_run_ids,
            "source_count": len(source_rows),
        },
    )
    summary_path = job.run_dir / "artifacts" / "design_campaign" / "workflow_summary.json"
    write_json(
        summary_path,
        {
            "campaign_name": clean_name,
            "target_key": target_key,
            "workflows": workflows,
            "candidate_count": len(normalized),
            "harmonized_candidate_count": len(normalized),
            "common_validation": {
                "profile": "design campaign collection",
                "native_candidate_count": metrics_total["native_candidate_count"],
                "common_bindcraft_passing_count": metrics_total["common_bindcraft_passing_count"],
            },
            "evaluation": {"status": "mixed"},
            "collection_sources": source_rows,
        },
    )
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": {
                "candidates": normalized,
                "workflow_summary": str(summary_path.relative_to(job.run_dir)),
                "collection": str(collection_path.relative_to(job.run_dir)),
                "design_campaign_collection_sources": str(sources_path.relative_to(job.run_dir)),
            },
            "metrics": {
                **metrics_total,
                "collection_name": clean_name,
                "source_run_count": len(source_rows),
                "candidate_count": len(normalized),
                "harmonized_candidate_count": len(normalized),
                "engine_count": len(
                    {
                        str(candidate.get("raw_metadata", {}).get("design_campaign_engine") or candidate.get("source_tool") or "")
                        for candidate in normalized
                    }
                    - {""}
                ),
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
                "collection": str(collection_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir
