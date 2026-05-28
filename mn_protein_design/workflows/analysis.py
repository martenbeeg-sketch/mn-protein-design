from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from mn_protein_design.core.candidates import (
    STAGE_ANALYSIS,
    STAGE_COMPLEX_REFOLDING,
    read_candidates,
    write_candidates,
)
from mn_protein_design.core.hotspot_metrics import calculate_hotspot_metrics
from mn_protein_design.core.jobs import create_job, finish_job, update_status, write_json


ANALYSIS_GROUP = "analysis"
IPSAE_RUNNER = Path("/home/user/programs/ovo-git/ovo/pipelines/ipsae-tool/bin/run_ipsae.py")
IPSAE_FULL_RUNNER = Path("tools_to_implement/de_novo_binder_scoring/scripts/ipsae_w_ipae.py")
IPSAE_IMAGE = "ovoex-ipsae:latest"


DEFAULT_THRESHOLDS = {
    "min_binder_plddt": 70.0,
    "min_confidence": 0.0,
    "min_iptm": 0.0,
    "min_ipsae": 0.0,
    "max_ipae": 10.0,
    "max_ipde": 20.0,
    "max_binder_rmsd": 5.0,
}

METRIC_ALIASES = {
    "binder_plddt": [
        "binder_plddt",
        "plddt",
        "boltz2_binder_plddt",
        "boltz2_complex_plddt",
        "boltz2_confidence_score",
    ],
    "confidence": [
        "confidence",
        "confidence_score",
        "boltz2_confidence_score",
        "boltz2_confidence",
    ],
    "iptm": [
        "iptm",
        "i_ptm",
        "boltz2_iptm",
        "boltz2_pair_chains_iptm",
    ],
    "ipae": [
        "ipae",
        "i_pae",
        "binder_pae",
        "boltz2_ipae",
    ],
    "ipde": [
        "ipde",
        "boltz2_ipde",
        "boltz2_complex_ipde",
    ],
    "binder_rmsd": [
        "target_aligned_binder_rmsd",
        "binder_rmsd",
        "boltz2_target_aligned_binder_rmsd",
        "design_backbone_rmsd",
    ],
    "monomer_rmsd": [
        "monomer_refolding_rmsd",
        "monomer_rmsd",
        "binder_monomer_rmsd",
    ],
    "ipsae": [
        "ipsae",
        "ipSAE",
        "ipSAE_min",
        "ipsae_min",
    ],
    "ipsae_min": [
        "ipsae_min",
        "ipSAE_min",
    ],
    "ipsae_max": [
        "ipsae_max",
        "ipSAE_max",
    ],
    "ipsae_avg": [
        "ipsae_avg",
        "ipSAE_avg",
    ],
    "lis": [
        "lis",
        "LIS",
    ],
    "ipsae_min_in_calculation": [
        "ipsae_min_in_calculation",
        "ipSAE_min_in_calculation",
    ],
    "ipsae_d0chn": [
        "ipsae_d0chn",
        "ipSAE_d0chn",
    ],
    "ipsae_d0dom": [
        "ipsae_d0dom",
        "ipSAE_d0dom",
    ],
    "pdockq_min": [
        "pdockq_min",
        "pDockQ_min",
    ],
    "pdockq_max": [
        "pdockq_max",
        "pDockQ_max",
    ],
    "pdockq2_min": [
        "pdockq2_min",
        "pDockQ2_min",
    ],
    "pdockq2_max": [
        "pdockq2_max",
        "pDockQ2_max",
    ],
    "hotspot_contact_fraction": ["hotspot_contact_fraction"],
    "hotspots_contacted": ["hotspots_contacted"],
    "hotspot_contact_pairs": ["hotspot_contact_pairs"],
    "min_binder_to_hotspot_distance": ["min_binder_to_hotspot_distance"],
    "mean_binder_to_hotspot_distance": ["mean_binder_to_hotspot_distance"],
    "binder_interface_contacts": ["binder_interface_contacts"],
    "hotspot_interface_contact_fraction": ["hotspot_interface_contact_fraction"],
    "binder_radius_of_gyration": ["binder_radius_of_gyration"],
    "binder_end_to_end_distance": ["binder_end_to_end_distance"],
}

BINDCRAFT_METRICS = [
    "Protocol",
    "Length",
    "Seed",
    "Helicity",
    "Target_Hotspot",
    "InterfaceResidues",
    "pAE",
    "i_pLDDT",
    "ss_pLDDT",
    "Unrelaxed_Clashes",
    "Relaxed_Clashes",
    "Binder_Energy_Score",
    "Surface_Hydrophobicity",
    "ShapeComplementarity",
    "PackStat",
    "dG",
    "dSASA",
    "dG/dSASA",
    "Interface_SASA_%",
    "Interface_Hydrophobicity",
    "n_InterfaceResidues",
    "n_InterfaceHbonds",
    "InterfaceHbondsPercentage",
    "n_InterfaceUnsatHbonds",
    "InterfaceUnsatHbondsPercentage",
    "Interface_Helix%",
    "Interface_BetaSheet%",
    "Interface_Loop%",
    "Binder_Helix%",
    "Binder_BetaSheet%",
    "Binder_Loop%",
    "InterfaceAAs",
    "Target_RMSD",
    "TrajectoryTime",
    "Notes",
]


def _candidate_artifacts(run_dir: Path) -> list[dict[str, str]]:
    artifacts = []
    for path in sorted((run_dir / "artifacts" / "analysis").glob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "name": path.stem,
                    "path": str(path.relative_to(run_dir)),
                    "type": "analysis_table" if path.suffix == ".csv" else "analysis_summary",
                }
            )
    for path in sorted((run_dir / "artifacts" / "normalized_candidates").glob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "name": path.stem,
                    "path": str(path.relative_to(run_dir)),
                    "type": "normalized_candidates" if path.suffix == ".jsonl" else "campaign_result",
                }
            )
    return artifacts


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _resolve_path(base_dir: Path, path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    return path if path.is_absolute() else base_dir / path


def _safe_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or "candidate").replace("/", "_").replace(" ", "_")


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        flat_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, flat_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[flat_key] = value
    return flat


def _find_pae_path(source_run_dir: Path, candidate: dict[str, Any], structure_path: Path | None) -> Path | None:
    search_dirs: list[Path] = []
    raw = candidate.get("raw_metadata") or {}
    explicit_pae = raw.get("pae_path") or candidate.get("pae_path")
    if explicit_pae:
        resolved = _resolve_path(source_run_dir, str(explicit_pae))
        if resolved and resolved.exists():
            return resolved
    if structure_path:
        direct_matches = [
            structure_path.with_name(f"{structure_path.stem}_pae.json"),
            structure_path.with_name(f"{structure_path.stem}.pae.json"),
            structure_path.with_name(f"pae_{structure_path.stem}.json"),
            structure_path.with_suffix(".npz"),
        ]
        for match in direct_matches:
            if match.exists():
                return match
    prediction_dir = raw.get("prediction_dir")
    if prediction_dir:
        resolved = _resolve_path(source_run_dir, str(prediction_dir))
        if resolved:
            search_dirs.append(resolved)
    if structure_path:
        search_dirs.append(structure_path.parent)
    search_dirs.append(source_run_dir)

    patterns = [
        "pae*.npz",
        "*pae*.npz",
        "pae*.json",
        "*pae*.json",
        "scores*.json",
        "*scores*.json",
    ]
    seen: set[Path] = set()
    for directory in search_dirs:
        if not directory or not directory.exists() or directory in seen:
            continue
        seen.add(directory)
        if structure_path:
            stem = structure_path.stem
            for pattern in [f"{stem}*pae*.json", f"{stem}*pae*.npz", f"*{stem}*pae*.json", f"*{stem}*pae*.npz"]:
                matches = sorted(directory.glob(pattern))
                if matches:
                    return matches[0]
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    return None


def _ipsae_engine(candidate: dict[str, Any], pae_path: Path | None) -> tuple[str, str]:
    source_tool = str(candidate.get("source_tool") or candidate.get("tool") or "").lower()
    if pae_path and pae_path.suffix.lower() == ".npz":
        return "boltz", "boltz_npz"
    if "boltz" in source_tool:
        return "boltz", "boltz_npz"
    return "af2", "af2_json"


def _parse_ipsae_row(path: Path) -> dict[str, Any]:
    header = [
        "ID",
        "ipsae",
        "ipsae_d0chn",
        "ipsae_d0dom",
        "iptm_af",
        "iptm_d0chn",
        "binder_chain",
        "target_chain",
        "type",
        "engine",
        "pae_format",
        "error",
    ]
    if not path.exists():
        return {"ipsae_error": "IPSAE output row missing"}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {"ipsae_error": "IPSAE output row empty"}
    row = dict(zip(header, rows[0]))
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        if key == "ID":
            continue
        if key == "error":
            parsed["ipsae_error"] = value
            continue
        number = _safe_float(value)
        parsed[key] = number if number is not None else value
    return parsed


def _find_ipsae_summary(structure: Path, pae_cutoff: float, dist_cutoff: float) -> Path | None:
    pae_text = f"{int(pae_cutoff):02d}"
    dist_text = f"{int(dist_cutoff):02d}"
    candidates = [
        structure.parent / f"{structure.stem}_{pae_text}_{dist_text}.txt",
        structure.parent / f"{structure.with_suffix('').name}_{pae_text}_{dist_text}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(
        [
            path
            for path in structure.parent.glob(f"{structure.stem}_*_*.txt")
            if not path.name.endswith("_byres.txt") and not path.name.endswith("_done.txt")
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _float_field(row: dict[str, str], key: str) -> float | None:
    return _safe_float(row.get(key))


def _parse_ipsae_summary_scores(summary_path: Path, focus_chain: str = "A") -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if not lines:
        return {"ipsae_error": "IPSAE summary file empty"}

    header_index = None
    headers: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("Chn1") and "Chn2" in line and "ipSAE" in line:
            header_index = index
            headers = line.split()
            break
    if header_index is None:
        return {"ipsae_error": f"could not parse IPSAE summary: {summary_path.name}"}

    rows: list[dict[str, str]] = []
    for line in lines[header_index + 1 :]:
        parts = line.split()
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, parts))
        if focus_chain in {row.get("Chn1"), row.get("Chn2")}:
            rows.append(row)
    if not rows:
        return {"ipsae_error": f"no IPSAE chain pairs found for chain {focus_chain}"}

    data_max: dict[str, dict[str, list[float]]] = {}
    data_asym: dict[str, list[float]] = {}
    ipae_by_partner: dict[str, list[float]] = {}
    pdockq_by_partner: dict[str, dict[str, list[float]]] = {}
    qualified_contact_rows = 0

    for row in rows:
        chain1 = row.get("Chn1")
        chain2 = row.get("Chn2")
        if not chain1 or not chain2 or focus_chain not in {chain1, chain2}:
            continue
        partner = chain2 if chain1 == focus_chain else chain1
        row_type = (row.get("Type") or "").lower()

        if row_type == "max":
            ipae = _float_field(row, "ipae")
            if ipae is not None:
                ipae_by_partner.setdefault(partner, []).append(ipae)

        dist1 = _float_field(row, "dist1")
        dist2 = _float_field(row, "dist2")
        has_qualified_contacts = dist1 not in {None, 0.0} and dist2 not in {None, 0.0}
        if has_qualified_contacts:
            qualified_contact_rows += 1

        p_bucket = pdockq_by_partner.setdefault(partner, {"pDockQ": [], "pDockQ2": []})
        for source_key in p_bucket:
            value = _float_field(row, source_key)
            if value is not None:
                p_bucket[source_key].append(value)

        if row_type == "max":
            bucket = data_max.setdefault(
                partner,
                {
                    "ipSAE": [],
                    "ipSAE_avg": [],
                    "LIS": [],
                    "ipSAE_min_in_calculation": [],
                    "ipSAE_d0chn": [],
                    "ipSAE_d0dom": [],
                },
            )
            for source_key in bucket:
                value = _float_field(row, source_key)
                if value is not None:
                    bucket[source_key].append(value)
        elif row_type == "asym":
            value = _float_field(row, "ipSAE")
            if value is not None:
                data_asym.setdefault(partner, []).append(value)

    def avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    ipsae_min = avg([min(values) for values in data_asym.values() if values])
    ipsae_max = avg([max(values["ipSAE"]) for values in data_max.values() if values["ipSAE"]])
    ipsae_avg = avg([min(values["ipSAE_avg"]) for values in data_max.values() if values["ipSAE_avg"]])
    lis = avg([min(values["LIS"]) for values in data_max.values() if values["LIS"]])
    ipsae_min_calc = avg(
        [
            min(values["ipSAE_min_in_calculation"])
            for values in data_max.values()
            if values["ipSAE_min_in_calculation"]
        ]
    )
    ipsae_d0chn = avg([min(values["ipSAE_d0chn"]) for values in data_max.values() if values["ipSAE_d0chn"]])
    ipsae_d0dom = avg([min(values["ipSAE_d0dom"]) for values in data_max.values() if values["ipSAE_d0dom"]])
    ipae = avg([min(values) for values in ipae_by_partner.values() if values])

    pdockq_min = avg([min(values["pDockQ"]) for values in pdockq_by_partner.values() if values["pDockQ"]])
    pdockq_max = avg([max(values["pDockQ"]) for values in pdockq_by_partner.values() if values["pDockQ"]])
    pdockq2_min = avg([min(values["pDockQ2"]) for values in pdockq_by_partner.values() if values["pDockQ2"]])
    pdockq2_max = avg([max(values["pDockQ2"]) for values in pdockq_by_partner.values() if values["pDockQ2"]])

    result = {
        "ipsae": ipsae_min,
        "ipsae_min": ipsae_min,
        "ipsae_max": ipsae_max,
        "ipsae_avg": ipsae_avg,
        "lis": lis,
        "ipsae_min_in_calculation": ipsae_min_calc,
        "ipsae_d0chn": ipsae_d0chn,
        "ipsae_d0dom": ipsae_d0dom,
        "ipae": ipae,
        "pdockq_min": pdockq_min,
        "pdockq_max": pdockq_max,
        "pdockq2_min": pdockq2_min,
        "pdockq2_max": pdockq2_max,
        "ipsae_summary": str(summary_path),
    }
    if ipsae_min is None and ipsae_max is None:
        result["ipsae_error"] = "IPSAE summary has no parsable chain-pair scores"
    elif qualified_contact_rows == 0:
        result["ipsae_error"] = "no PAE-qualified interface contacts at current IPSAE cutoffs"
    return result


def _pae_matrix_from_json(path: Path) -> list[list[float]] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    matrix = payload.get("predicted_aligned_error") or payload.get("pae")
    if not isinstance(matrix, list) or not matrix:
        return None
    rows: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list):
            return None
        values = [_safe_float(value) for value in row]
        if any(value is None for value in values):
            return None
        rows.append([float(value) for value in values if value is not None])
    if any(len(row) != len(rows) for row in rows):
        return None
    return rows


def _chain_ca_coordinates(structure_path: Path) -> dict[str, list[tuple[float, float, float]]]:
    chains: dict[str, list[tuple[float, float, float]]] = {}
    seen: set[tuple[str, str, str]] = set()
    with structure_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or line[12:16].strip() != "CA":
                continue
            chain = line[21].strip()
            residue_key = (chain, line[22:26].strip(), line[26].strip())
            if residue_key in seen:
                continue
            seen.add(residue_key)
            try:
                coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            chains.setdefault(chain, []).append(coord)
    return chains


def _chain_roles(candidate: dict[str, Any]) -> tuple[str, str]:
    binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
    target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
    return (binder_chains[0] if binder_chains else "A", target_chains[0] if target_chains else "B")


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _interface_pae_scores(
    structure_path: Path | None,
    pae_path: Path | None,
    binder_chain: str = "A",
    target_chain: str = "B",
    dist_cutoff: float = 10.0,
) -> dict[str, Any]:
    if not structure_path or not structure_path.exists() or not pae_path or not pae_path.exists() or pae_path.suffix.lower() != ".json":
        return {}
    matrix = _pae_matrix_from_json(pae_path)
    if matrix is None:
        return {}
    chains = _chain_ca_coordinates(structure_path)
    binder = chains.get(binder_chain) or []
    target = chains.get(target_chain) or []
    if not binder or not target:
        return {}
    if len(matrix) < len(binder) + len(target):
        return {}

    offsets: dict[str, int] = {}
    offset = 0
    for chain, coords in chains.items():
        offsets[chain] = offset
        offset += len(coords)
    binder_offset = offsets.get(binder_chain, 0)
    target_offset = offsets.get(target_chain, len(binder))
    pairs: list[tuple[int, int]] = []
    for binder_index, binder_coord in enumerate(binder):
        for target_index, target_coord in enumerate(target):
            if _distance(binder_coord, target_coord) <= dist_cutoff:
                pairs.append((binder_index, target_index))
    if not pairs:
        return {}

    binder_to_target = [matrix[binder_offset + binder_index][target_offset + target_index] for binder_index, target_index in pairs]
    target_to_binder = [matrix[target_offset + target_index][binder_offset + binder_index] for binder_index, target_index in pairs]
    combined = binder_to_target + target_to_binder
    return {
        "ipae": sum(combined) / len(combined),
        "ipae_binder_to_target": sum(binder_to_target) / len(binder_to_target),
        "ipae_target_to_binder": sum(target_to_binder) / len(target_to_binder),
        "ipae_contact_pairs": len(pairs),
        "ipae_dist_cutoff": dist_cutoff,
    }


def _run_ipsae(source_run_dir: Path, run_dir: Path, candidates: list[dict[str, Any]], pae_cutoff: float = 10.0, dist_cutoff: float = 10.0) -> list[dict[str, Any]]:
    ipsae_dir = run_dir / "artifacts" / "analysis" / "ipsae"
    input_dir = ipsae_dir / "inputs"
    output_dir = ipsae_dir / "rows"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    enhanced: list[dict[str, Any]] = []

    for candidate in candidates:
        safe_id = _safe_id(candidate)
        metrics = dict(candidate.get("metrics") or {})
        binder_chain, target_chain = _chain_roles(candidate)
        structure_path = _resolve_path(source_run_dir, candidate.get("complex_pdb"))
        pae_path = _find_pae_path(source_run_dir, candidate, structure_path)
        staged_dir = input_dir / safe_id
        staged_dir.mkdir(parents=True, exist_ok=True)
        row_path = output_dir / f"{safe_id}_ipsae_row.csv"

        if not structure_path or not structure_path.exists():
            metrics["ipsae_error"] = "complex structure missing"
            enhanced.append({**candidate, "metrics": metrics})
            continue
        if not pae_path or not pae_path.exists():
            metrics["ipsae_error"] = "PAE file missing"
            enhanced.append({**candidate, "metrics": metrics})
            continue

        staged_structure = staged_dir / structure_path.name
        staged_pae = staged_dir / pae_path.name
        shutil.copy2(structure_path, staged_structure)
        shutil.copy2(pae_path, staged_pae)
        engine, pae_format = _ipsae_engine(candidate, staged_pae)
        full_runner = Path(IPSAE_FULL_RUNNER)
        if full_runner.exists() and staged_pae.suffix.lower() == ".json" and staged_structure.suffix.lower() == ".pdb":
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{run_dir}:/work",
                "-v",
                f"{full_runner.resolve().parent}:/scripts:ro",
                "-w",
                "/work",
                IPSAE_IMAGE,
                "python3",
                f"/scripts/{full_runner.name}",
                f"/work/artifacts/analysis/ipsae/inputs/{safe_id}/{staged_pae.name}",
                f"/work/artifacts/analysis/ipsae/inputs/{safe_id}/{staged_structure.name}",
                str(pae_cutoff),
                str(dist_cutoff),
            ]
        else:
            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{run_dir}:/work",
                "-v",
                f"{IPSAE_RUNNER.parent}:/scripts:ro",
                "-w",
                "/work",
                IPSAE_IMAGE,
                "python3",
                "/scripts/run_ipsae.py",
                "--id",
                safe_id,
                "--structure",
                f"/work/artifacts/analysis/ipsae/inputs/{safe_id}/{staged_structure.name}",
                "--pae",
                f"/work/artifacts/analysis/ipsae/inputs/{safe_id}/{staged_pae.name}",
                "--pae-format",
                pae_format,
                "--engine",
                engine,
                "--binder-chain",
                binder_chain,
                "--target-chain",
                target_chain,
                "--pae-cutoff",
                str(pae_cutoff),
                "--dist-cutoff",
                str(dist_cutoff),
                "--output",
                f"/work/artifacts/analysis/ipsae/rows/{safe_id}_ipsae_row.csv",
            ]
        steps.append({"name": f"ipsae-{safe_id}", "command": command, "row_path": str(row_path)})
        enhanced.append({**candidate, "metrics": metrics})

    if steps:
        command_payload = {
            "mode": "internal+docker",
            "steps": steps,
            "note": "Internal ranking/filter analysis with mandatory IPSAE attempts for every candidate with structure and PAE files.",
        }
        write_json(run_dir / "command.json", command_payload)
        with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
            for step in steps:
                stdout.write(f"$ {' '.join(step['command'])}\n")
                stdout.flush()
                proc = subprocess.run(step["command"], stdout=stdout, stderr=stderr, check=False)
                if proc.returncode != 0:
                    stderr.write(f"IPSAE step {step['name']} exited with {proc.returncode}\n")

    by_id = {str(candidate.get("candidate_id")): candidate for candidate in enhanced}
    for candidate in enhanced:
        safe_id = _safe_id(candidate)
        row_path = output_dir / f"{safe_id}_ipsae_row.csv"
        staged_structure = None
        structure_path = _resolve_path(source_run_dir, candidate.get("complex_pdb"))
        pae_path = _find_pae_path(source_run_dir, candidate, structure_path)
        if structure_path:
            staged_structure = input_dir / safe_id / structure_path.name
        row_metrics = _parse_ipsae_row(row_path) if row_path.exists() else {}
        summary_metrics: dict[str, Any] = {}
        if staged_structure and staged_structure.exists():
            summary_path = _find_ipsae_summary(staged_structure, pae_cutoff, dist_cutoff)
            if summary_path:
                binder_chain, target_chain = _chain_roles(candidate)
                summary_metrics = _parse_ipsae_summary_scores(summary_path, focus_chain=binder_chain)
        if row_metrics or summary_metrics:
            binder_chain, target_chain = _chain_roles(candidate)
            pae_metrics = _interface_pae_scores(
                structure_path,
                pae_path,
                binder_chain=binder_chain,
                target_chain=target_chain,
                dist_cutoff=dist_cutoff,
            )
            candidate["metrics"] = {**dict(candidate.get("metrics") or {}), **pae_metrics, **row_metrics, **summary_metrics}
        by_id[str(candidate.get("candidate_id"))] = candidate
    return enhanced


def _metric_value(flat_metrics: dict[str, Any], canonical: str) -> float | None:
    aliases = METRIC_ALIASES.get(canonical, [canonical])
    lowered = {key.lower(): value for key, value in flat_metrics.items()}
    for alias in aliases:
        if alias.lower() in lowered:
            value = _safe_float(lowered[alias.lower()])
            if value is not None:
                return value
    for key, value in lowered.items():
        if any(key.endswith(f".{alias.lower()}") for alias in aliases):
            number = _safe_float(value)
            if number is not None:
                return number
    return None


def _add_hotspot_metrics(source_run_dir: Path, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enhanced: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = dict(candidate.get("metrics") or {})
        structure_path = _resolve_path(source_run_dir, candidate.get("complex_pdb"))
        if structure_path and structure_path.exists() and structure_path.suffix.lower() == ".pdb":
            binder_chain, target_chain = _chain_roles(candidate)
            hotspot_metrics = calculate_hotspot_metrics(
                structure_path,
                candidate.get("hotspots") or [],
                binder_chain=str(metrics.get("binder_chain") or binder_chain),
                target_chain=str(metrics.get("target_chain") or target_chain),
                contact_cutoff=8.0,
                contig=str(candidate.get("contig") or ""),
            )
            metrics.update(hotspot_metrics)
        enhanced.append({**candidate, "metrics": metrics})
    return enhanced


def _score_row(row: dict[str, Any]) -> float:
    score = 0.0
    if row.get("binder_plddt") is not None:
        score += min(max(float(row["binder_plddt"]), 0.0), 100.0) / 100.0
    if row.get("confidence") is not None:
        confidence = float(row["confidence"])
        score += confidence if confidence <= 1.5 else confidence / 100.0
    if row.get("iptm") is not None:
        score += float(row["iptm"])
    if row.get("ipsae") is not None:
        score += float(row["ipsae"])
    if row.get("ipae") is not None:
        score += max(0.0, 1.0 - float(row["ipae"]) / 30.0)
    if row.get("ipde") is not None:
        score += max(0.0, 1.0 - float(row["ipde"]) / 30.0)
    if row.get("binder_rmsd") is not None:
        score += max(0.0, 1.0 - float(row["binder_rmsd"]) / 10.0)
    if row.get("monomer_rmsd") is not None:
        score += max(0.0, 1.0 - float(row["monomer_rmsd"]) / 10.0)
    if row.get("hotspot_contact_fraction") is not None:
        score += min(1.0, max(0.0, float(row["hotspot_contact_fraction"])))
    return round(score, 4)


def _passes_thresholds(row: dict[str, Any], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = [
        ("binder_plddt", ">=", thresholds.get("min_binder_plddt")),
        ("confidence", ">=", thresholds.get("min_confidence")),
        ("iptm", ">=", thresholds.get("min_iptm")),
        ("ipsae", ">=", thresholds.get("min_ipsae")),
        ("ipae", "<=", thresholds.get("max_ipae")),
        ("ipde", "<=", thresholds.get("max_ipde")),
        ("binder_rmsd", "<=", thresholds.get("max_binder_rmsd")),
        ("monomer_rmsd", "<=", thresholds.get("max_binder_rmsd")),
        ("hotspot_contact_fraction", ">=", thresholds.get("min_hotspot_contact_fraction")),
        ("min_binder_to_hotspot_distance", "<=", thresholds.get("max_hotspot_distance")),
    ]
    for metric, op, threshold in checks:
        if threshold is None:
            continue
        value = row.get(metric)
        if value is None:
            continue
        if op == ">=" and float(value) < float(threshold):
            failures.append(f"{metric} < {threshold:g}")
        if op == "<=" and float(value) > float(threshold):
            failures.append(f"{metric} > {threshold:g}")
    return not failures, failures


def build_analysis_table(candidates: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows: list[dict[str, Any]] = []
    for source_index, candidate in enumerate(candidates):
        flat_metrics = _flatten_metrics(candidate.get("metrics") or {})
        row = {
            "source_index": source_index,
            "candidate_id": candidate.get("candidate_id"),
            "source_tool": candidate.get("source_tool"),
            "complex_pdb": candidate.get("complex_pdb"),
            "binder_pdb": candidate.get("binder_pdb"),
            "binder_sequence": candidate.get("binder_sequence"),
            "parent_count": len(candidate.get("parents") or []),
            "metric_count": len(flat_metrics),
        }
        for metric in METRIC_ALIASES:
            if metric == "ipde":
                backend = str(flat_metrics.get("complex_refolding_backend") or "").lower()
                row[metric] = _metric_value(flat_metrics, metric) if "boltz" in backend else None
            else:
                row[metric] = _metric_value(flat_metrics, metric)
            if metric == "binder_plddt" and row[metric] is not None and float(row[metric]) <= 1.5:
                row[metric] = float(row[metric]) * 100.0
        for metric in BINDCRAFT_METRICS:
            if metric in flat_metrics:
                row[f"bindcraft_{metric}"] = flat_metrics[metric]
        for metric, value in flat_metrics.items():
            if metric in row:
                metric = f"source_{metric}"
            row[metric] = value
        native_pass_filters = flat_metrics.get("pass_filters")
        row["native_pass_filters"] = native_pass_filters if isinstance(native_pass_filters, bool) else None
        row["ipsae_error"] = str(flat_metrics.get("ipsae_error") or "")
        row["analysis_score"] = _score_row(row)
        passed, failures = _passes_thresholds(row, thresholds)
        row["passes_filters"] = passed
        row["filter_failures"] = "; ".join(failures)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["passes_filters", "analysis_score"], ascending=[False, False]).reset_index(drop=True)


def _summary_payload(df: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "thresholds": thresholds,
        "candidate_count": int(len(df)),
        "passing_count": int(df["passes_filters"].sum()) if "passes_filters" in df else 0,
        "ipsae_error_count": int(df["ipsae_error"].fillna("").astype(bool).sum()) if "ipsae_error" in df else 0,
    }
    numeric_cols = [col for col in METRIC_ALIASES if col in df.columns]
    for col in ["analysis_score", *numeric_cols]:
        if col not in df:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary[col] = {
            "min": float(series.min()),
            "median": float(series.median()),
            "max": float(series.max()),
        }
    return summary


def run_analysis_contract(
    source_run_dir: Path,
    candidates_jsonl: Path,
    tool: str = "ranking",
    keep_top_n: int = 100,
    thresholds: dict[str, float] | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = [
        candidate
        for candidate in read_candidates(candidates_jsonl)
        if candidate.get("stage") == STAGE_COMPLEX_REFOLDING
    ]
    if not source_candidates:
        raise ValueError("No complex_refolding candidates were found in the selected candidate set.")

    job = create_job(
        ANALYSIS_GROUP,
        job_type="analysis",
        tool=tool,
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={"keep_top_n": keep_top_n, "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})}, "backend": "internal"},
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "internal",
            "command": [],
            "note": "Internal ranking/filter analysis over normalized complex-refolding candidate metrics.",
        },
    )
    update_status(job.run_dir, "running")
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    analysis_dir = job.run_dir / "artifacts" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    source_candidates = _run_ipsae(source_run_dir, job.run_dir, source_candidates)
    source_candidates = _add_hotspot_metrics(source_run_dir, source_candidates)
    df = build_analysis_table(source_candidates, thresholds)
    df.insert(0, "analysis_rank", range(1, len(df) + 1))
    ranked_csv = analysis_dir / "ranked_candidates.csv"
    pass_fail_csv = analysis_dir / "pass_fail_counts.csv"
    summary_json = analysis_dir / "metrics_summary.json"
    df.to_csv(ranked_csv, index=False)
    if "passes_filters" in df:
        df["passes_filters"].value_counts(dropna=False).rename_axis("passes_filters").reset_index(name="count").to_csv(pass_fail_csv, index=False)
    summary = _summary_payload(df, thresholds)
    write_json(summary_json, summary)

    ranked = source_candidates if df.empty else [
        source_candidates[int(source_index)]
        for source_index in df["source_index"].head(max(1, int(keep_top_n))).tolist()
    ]
    ranked_rows = df.head(max(1, int(keep_top_n))).to_dict(orient="records") if not df.empty else []
    candidates: list[dict[str, Any]] = []
    for rank, source in enumerate(ranked, start=1):
        row = ranked_rows[rank - 1] if rank - 1 < len(ranked_rows) else {}
        metrics = dict(source.get("metrics") or {})
        metrics.update(
            {
                "analysis_rank": rank,
                "analysis_score": row.get("analysis_score"),
                "passes_filters": row.get("passes_filters"),
                "filter_failures": row.get("filter_failures"),
                "analysis_backend": "internal",
                "result_kind": "app_reanalysis",
            }
        )
        candidates.append(
            {
                **source,
                "candidate_id": f"{source.get('candidate_id')}_{tool}_{rank:03d}",
                "stage": STAGE_ANALYSIS,
                "source_tool": tool,
                "tool": tool,
                "metrics": metrics,
                "parents": [str(source.get("candidate_id") or "")],
                "raw_metadata": {
                    **dict(source.get("raw_metadata") or {}),
                    "result_kind": "app_reanalysis",
                    "source_candidate": source,
                    "backend_status": "internal",
                },
            }
        )

    normalized = write_candidates(job.run_dir, tool, candidates)
    artifacts = _candidate_artifacts(job.run_dir)
    finish_job(
        job.run_dir,
        bool(normalized),
        {
            "outputs": {"artifacts": artifacts, "candidates": normalized},
            "metrics": {
                "candidate_count": len(normalized),
                "passing_count": summary.get("passing_count", 0),
                "backend_status": "internal",
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
            "summary": summary,
        },
    )
    return job.run_dir
