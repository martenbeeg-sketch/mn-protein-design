from __future__ import annotations

import json
import gzip
import math
import shlex
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from mn_protein_design.core.hotspot_metrics import (
    calculate_hotspot_atom_contact_metrics,
    calculate_hotspot_ca_contact_metrics,
    calculate_reference_aligned_hotspot_atom_contact_metrics,
)
from mn_protein_design.core.jobs import write_json
from mn_protein_design.core.portable_paths import (
    is_portable_path,
    portable_path,
    resolve_managed_paths,
    resolve_stored_path,
    store_managed_paths,
)


CANDIDATE_SCHEMA_VERSION = "mn-protein-design.candidate.v1"


STAGE_GENERATION_BACKBONE = "generation.backbone"
STAGE_GENERATION_BACKBONE_SEQUENCE = "generation.backbone_sequence"
STAGE_SEQUENCE_DESIGN = "sequence_design"
STAGE_MONOMER_REFOLDING = "monomer_refolding"
STAGE_COMPLEX_REFOLDING = "complex_refolding"
STAGE_ANALYSIS = "analysis"
STAGE_BENCHMARK = "benchmark"

LEGACY_STAGE_MAP = {
    "backbone": STAGE_GENERATION_BACKBONE,
    "trajectory": STAGE_GENERATION_BACKBONE,
    "backbone+sequence": STAGE_GENERATION_BACKBONE_SEQUENCE,
    "backbone+validation": STAGE_COMPLEX_REFOLDING,
    "backbone+sequence+refolding": STAGE_COMPLEX_REFOLDING,
    "design_only": STAGE_GENERATION_BACKBONE,
}


@dataclass
class Candidate:
    candidate_id: str
    stage: str
    source_tool: str
    target_pdb: str | None = None
    complex_pdb: str | None = None
    binder_pdb: str | None = None
    binder_sequence: str | None = None
    target_chains: list[str] = field(default_factory=list)
    binder_chains: list[str] = field(default_factory=list)
    hotspots: list[str] = field(default_factory=list)
    binder_length: str | None = None
    contig: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def normalize_stage(stage: object) -> str:
    text = str(stage or "").strip()
    return LEGACY_STAGE_MAP.get(text, text or STAGE_GENERATION_BACKBONE)


def normalize_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    source_tool = str(payload.get("source_tool") or payload.get("tool") or "")
    binder_sequence = payload.get("binder_sequence")
    if binder_sequence is None:
        binder_sequence = payload.get("sequence")
    normalized = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": str(payload.get("candidate_id") or ""),
        "stage": normalize_stage(payload.get("stage")),
        "source_tool": source_tool,
        "tool": source_tool,
        "target_pdb": payload.get("target_pdb"),
        "complex_pdb": payload.get("complex_pdb"),
        "binder_pdb": payload.get("binder_pdb"),
        "binder_sequence": binder_sequence,
        "target_chains": list(payload.get("target_chains") or []),
        "binder_chains": list(payload.get("binder_chains") or []),
        "hotspots": list(payload.get("hotspots") or []),
        "binder_length": payload.get("binder_length"),
        "contig": payload.get("contig"),
        "metrics": dict(payload.get("metrics") or {}),
        "parents": list(payload.get("parents") or []),
        "raw_metadata": dict(payload.get("raw_metadata") or {}),
    }
    if payload.get("tool") and not normalized["raw_metadata"].get("legacy_tool"):
        normalized["raw_metadata"]["legacy_tool"] = payload.get("tool")
    if payload.get("sequence") and normalized["binder_sequence"] == payload.get("sequence"):
        normalized["raw_metadata"].setdefault("legacy_sequence_key", "sequence")
    return normalized


def candidates_dir(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "normalized_candidates"


def candidates_jsonl_path(run_dir: Path) -> Path:
    return candidates_dir(run_dir) / "candidates.jsonl"


def campaign_result_path(run_dir: Path) -> Path:
    return candidates_dir(run_dir) / "campaign_result.json"


def _candidate_roots(run_dir: Path, candidate: dict[str, Any]) -> list[Path]:
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source_run_dir = None
    source_run_dir_text = raw_metadata.get("source_run_dir")
    if source_run_dir_text:
        source_run_dir = resolve_stored_path(source_run_dir_text, must_exist=True)
    roots = [run_dir]
    if source_run_dir is not None:
        roots.insert(0, source_run_dir)
    return roots


def _resolve_candidate_path(run_dir: Path, value: object, candidate: dict[str, Any]) -> Path | None:
    if not value:
        return None
    for root in _candidate_roots(run_dir, candidate):
        resolved = resolve_stored_path(value, run_dir=root, must_exist=True)
        if resolved is not None:
            return resolved
    return None


def _resolve_candidate_structure_path(run_dir: Path, candidate: dict[str, Any]) -> Path | None:
    for key in ("complex_pdb", "binder_pdb"):
        resolved = _resolve_candidate_path(run_dir, candidate.get(key), candidate)
        if resolved is not None:
            return resolved
    return None


def _rel_path(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _structure_path_is_cif(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz"))


def _read_structure_text(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            magic = handle.read(2)
    except OSError:
        magic = b""
    if path.name.lower().endswith(".gz") or magic == b"\x1f\x8b":
        with gzip.open(path, "rt", errors="ignore") as handle:
            return handle.read()
    return path.read_text(errors="ignore")


def _mmcif_text_with_default_occupancy(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    atom_site_headers: list[str] = []
    in_atom_site_loop = False
    inserted_header = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_loop = True
            atom_site_headers = []
            inserted_header = False
            output.append(line)
            continue
        if in_atom_site_loop and stripped.startswith("_atom_site."):
            if stripped not in atom_site_headers:
                atom_site_headers.append(stripped)
            output.append(line)
            if stripped == "_atom_site.B_iso_or_equiv" and "_atom_site.occupancy" not in atom_site_headers:
                output.append("_atom_site.occupancy")
                atom_site_headers.append("_atom_site.occupancy")
                inserted_header = True
            continue
        if in_atom_site_loop and stripped.startswith(("ATOM ", "HETATM ")):
            tokens = shlex.split(stripped, posix=False)
            if inserted_header and len(tokens) == len(atom_site_headers) - 1:
                tokens.insert(atom_site_headers.index("_atom_site.occupancy"), "1.00")
                output.append(" ".join(tokens))
            else:
                output.append(line)
            continue
        if in_atom_site_loop and stripped.startswith("#"):
            in_atom_site_loop = False
            atom_site_headers = []
            inserted_header = False
        output.append(line)
    return "\n".join(output) + "\n"


def _parse_structure(path: Path) -> object:
    from Bio.PDB import MMCIFParser, PDBParser

    text = _read_structure_text(path)
    if _structure_path_is_cif(path):
        parser = MMCIFParser(QUIET=True)
        try:
            return parser.get_structure("candidate", StringIO(text))
        except KeyError as exc:
            if str(exc).strip("'\"") != "_atom_site.occupancy":
                raise
            return parser.get_structure("candidate", StringIO(_mmcif_text_with_default_occupancy(text)))
    return PDBParser(QUIET=True).get_structure("candidate", StringIO(text))


def _first_model(structure: object) -> object | None:
    return next(structure.get_models(), None)


def _safe_candidate_token(value: object) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "candidate"))
    return token.strip("._") or "candidate"


def _poly_gly_pdb_text_from_structure(path: Path) -> str:
    text = _read_structure_text(path)
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            if line[12:16].strip().upper() != "CA":
                continue
            lines.append("ATOM  " + line[6:17] + "GLY" + line[20:])
        elif line.startswith(("TER", "END")):
            lines.append(line)
    if not any(line.startswith("END") for line in lines):
        lines.append("END")
    return "\n".join(lines) + "\n"


def _write_poly_gly_backbone_reconstruction(
    run_dir: Path,
    candidate: dict[str, Any],
    structure_path: Path,
) -> tuple[Path | None, str]:
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except Exception as exc:
        return None, f"pdbfixer unavailable: {type(exc).__name__}: {exc}"

    out_dir = candidates_dir(run_dir) / "reconstructed_backbones"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_candidate_token(candidate.get('candidate_id'))}.poly_gly_backbone.pdb"
    try:
        fixer = PDBFixer(pdbfile=StringIO(_poly_gly_pdb_text_from_structure(structure_path)))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        with out_path.open("w") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    except Exception as exc:
        return None, f"reconstruction failed: {type(exc).__name__}: {exc}"
    return out_path, "created"


def _unit_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-6:
        return (1.0, 0.0, 0.0)
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _cross_vector(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _add_vector(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    return tuple(origin[index] + direction[index] * scale for index in range(3))  # type: ignore[return-value]


def _backbone_coords_from_ca_trace(ca_coords: list[tuple[float, float, float]]) -> list[list[np.ndarray]]:
    backbone: list[list[np.ndarray]] = []
    for index, ca in enumerate(ca_coords):
        if index + 1 < len(ca_coords):
            next_ca = ca_coords[index + 1]
            direction = _unit_vector(tuple(next_ca[axis] - ca[axis] for axis in range(3)))  # type: ignore[arg-type]
        elif index > 0:
            prev_ca = ca_coords[index - 1]
            direction = _unit_vector(tuple(ca[axis] - prev_ca[axis] for axis in range(3)))  # type: ignore[arg-type]
        else:
            direction = (1.0, 0.0, 0.0)
        normal = _unit_vector(_cross_vector(direction, (0.0, 0.0, 1.0)))
        if normal == (1.0, 0.0, 0.0):
            normal = _unit_vector(_cross_vector(direction, (0.0, 1.0, 0.0)))
        c_coord = _add_vector(ca, direction, 1.52)
        backbone.append(
            [
                np.asarray(_add_vector(ca, direction, -1.45), dtype=np.float32),
                np.asarray(ca, dtype=np.float32),
                np.asarray(c_coord, dtype=np.float32),
                np.asarray(_add_vector(c_coord, normal, 1.23), dtype=np.float32),
            ]
        )
    return backbone


def _pydssp_chain_assignments(pydssp: object, coords: list[list[np.ndarray]]) -> list[str]:
    if len(coords) < 4:
        return []
    chain_assignment = np.asarray(pydssp.assign(np.asarray(coords, dtype=np.float32), out_type="c3")).reshape(-1)
    return [str(value) for value in chain_assignment]


def _is_heavy_atom(atom: object) -> bool:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element != "H"
    return not str(getattr(atom, "name", "") or "").strip().upper().startswith("H")


def _chain_ca_coords(model: object, chain_ids: list[str]) -> list[np.ndarray]:
    coords: list[np.ndarray] = []
    for chain_id in chain_ids:
        chain_key = str(chain_id)
        if chain_key not in model:
            continue
        for residue in model[chain_key]:
            if getattr(residue, "id", ("", None, ""))[0] != " " or "CA" not in residue:
                continue
            coords.append(np.asarray(residue["CA"].coord, dtype=float))
    return coords


def _superpose_rmsd(fixed_coords: list[np.ndarray], moving_coords: list[np.ndarray]) -> tuple[float, int] | None:
    atom_count = min(len(fixed_coords), len(moving_coords))
    if atom_count < 3:
        return None
    fixed = np.asarray(fixed_coords[:atom_count], dtype=float)
    moving = np.asarray(moving_coords[:atom_count], dtype=float)
    fixed_center = fixed.mean(axis=0)
    moving_center = moving.mean(axis=0)
    fixed_centered = fixed - fixed_center
    moving_centered = moving - moving_center
    covariance = moving_centered.T @ fixed_centered
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        adjusted_vt = vt_matrix.copy()
        adjusted_vt[-1, :] *= -1
        rotation = adjusted_vt.T @ u_matrix.T
    translation = fixed_center - moving_center @ rotation
    transformed = moving @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((transformed - fixed) ** 2, axis=1))))
    return rmsd, atom_count


def _candidate_binder_geometry_metrics(structure_path: Path, binder_chains: list[str]) -> dict[str, Any]:
    if not binder_chains:
        return {}
    try:
        model = _first_model(_parse_structure(structure_path))
    except Exception as exc:
        return {"binder_geometry_status": f"structure unreadable: {type(exc).__name__}: {exc}"}
    if model is None:
        return {"binder_geometry_status": "no model"}

    heavy_coords: list[np.ndarray] = []
    ca_coords: list[np.ndarray] = []
    residue_ids: set[tuple[str, int, str]] = set()
    for chain_id in binder_chains:
        chain_key = str(chain_id)
        if chain_key not in model:
            continue
        for residue in model[chain_key]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] != " ":
                continue
            residue_has_heavy_atom = False
            for atom in residue:
                if not _is_heavy_atom(atom):
                    continue
                heavy_coords.append(np.asarray(atom.coord, dtype=float))
                residue_has_heavy_atom = True
            if "CA" in residue:
                ca_coords.append(np.asarray(residue["CA"].coord, dtype=float))
            if residue_has_heavy_atom:
                residue_ids.add((chain_key, int(residue_id[1]), str(residue_id[2]).strip()))

    metrics: dict[str, Any] = {
        "binder_geometry_status": "ok" if heavy_coords or ca_coords else "no binder coordinates",
        "binder_heavy_atom_count": int(len(heavy_coords)),
        "binder_residue_count": int(len(residue_ids)),
        "binder_ca_count": int(len(ca_coords)),
        "binder_radius_of_gyration": None,
        "binder_ca_radius_of_gyration": None,
        "binder_ca_end_to_end_distance": None,
    }
    if heavy_coords:
        heavy_array = np.asarray(heavy_coords, dtype=float)
        center = heavy_array.mean(axis=0)
        metrics["binder_radius_of_gyration"] = float(np.sqrt(np.mean(np.sum((heavy_array - center) ** 2, axis=1))))
    if ca_coords:
        ca_array = np.asarray(ca_coords, dtype=float)
        center = ca_array.mean(axis=0)
        metrics["binder_ca_radius_of_gyration"] = float(np.sqrt(np.mean(np.sum((ca_array - center) ** 2, axis=1))))
        metrics["binder_ca_end_to_end_distance"] = (
            float(np.linalg.norm(ca_array[-1] - ca_array[0])) if len(ca_array) > 1 else 0.0
        )
        metrics["binder_end_to_end_distance"] = metrics["binder_ca_end_to_end_distance"]
    return metrics


def _generation_target_alignment_metrics(
    structure_path: Path,
    reference_target_path: Path,
    target_chains: list[str],
    reference_target_chains: list[str],
) -> dict[str, Any]:
    if not target_chains or not reference_target_chains:
        return {"generation_target_alignment_status": "missing target chain metadata"}
    try:
        moving_model = _first_model(_parse_structure(structure_path))
        fixed_model = _first_model(_parse_structure(reference_target_path))
    except Exception as exc:
        return {"generation_target_alignment_status": f"structure unreadable: {type(exc).__name__}: {exc}"}
    if moving_model is None or fixed_model is None:
        return {"generation_target_alignment_status": "missing model"}
    transform = _superpose_rmsd(
        _chain_ca_coords(fixed_model, reference_target_chains),
        _chain_ca_coords(moving_model, target_chains),
    )
    if transform is None:
        return {"generation_target_alignment_status": "fewer than 3 matching target CA atoms"}
    rmsd, atom_count = transform
    return {
        "generation_target_alignment_status": "ok",
        "generation_target_alignment_rmsd": float(rmsd),
        "generation_target_alignment_ca_count": int(atom_count),
    }


MIN_HELIX_ELEMENT_LENGTH = 4
MIN_SHEET_ELEMENT_LENGTH = 3
MIN_COIL_SEPARATOR_LENGTH = 3


def _raw_secondary_structure_segments(assignments: list[str]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_state = ""
    start = 0
    for index, state in enumerate(assignments):
        normalized_state = state if state in {"H", "E"} else "-"
        if index == 0:
            current_state = normalized_state
            start = 1
            continue
        if normalized_state == current_state:
            continue
        segments.append({"state": current_state, "start": start, "end": index, "length": index - start + 1})
        current_state = normalized_state
        start = index + 1
    if assignments:
        segments.append(
            {
                "state": current_state,
                "start": start,
                "end": len(assignments),
                "length": len(assignments) - start + 1,
            }
        )
    return segments


def _stringent_secondary_structure_segments(assignments: list[str]) -> list[dict[str, Any]]:
    raw_segments = _raw_secondary_structure_segments(assignments)
    elements: list[dict[str, Any]] = []
    for segment in raw_segments:
        state = str(segment["state"])
        if state not in {"H", "E"}:
            continue
        min_length = MIN_HELIX_ELEMENT_LENGTH if state == "H" else MIN_SHEET_ELEMENT_LENGTH
        if int(segment["length"]) < min_length:
            continue
        if (
            elements
            and str(elements[-1]["state"]) == state
            and int(segment["start"]) - int(elements[-1]["end"]) - 1 < MIN_COIL_SEPARATOR_LENGTH
        ):
            elements[-1]["end"] = segment["end"]
            elements[-1]["length"] = int(elements[-1]["end"]) - int(elements[-1]["start"]) + 1
        else:
            elements.append(dict(segment))
    return elements


def _angle_degrees(
    left: tuple[float, float, float],
    center: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    first = np.asarray(left, dtype=float) - np.asarray(center, dtype=float)
    second = np.asarray(right, dtype=float) - np.asarray(center, dtype=float)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return 180.0
    cosine = float(np.dot(first, second) / denominator)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def _ca_trace_assignments(ca_coords: list[tuple[float, float, float]]) -> list[str]:
    assignments = ["-" for _ in ca_coords]
    for index in range(1, max(1, len(ca_coords) - 1)):
        angle = _angle_degrees(ca_coords[index - 1], ca_coords[index], ca_coords[index + 1])
        helix_distance = None
        if index + 3 < len(ca_coords):
            helix_distance = float(np.linalg.norm(np.asarray(ca_coords[index + 3]) - np.asarray(ca_coords[index])))
        if 75.0 <= angle <= 105.0 and helix_distance is not None and 4.5 <= helix_distance <= 6.7:
            assignments[index] = "H"
        elif angle >= 105.0:
            assignments[index] = "E"
    return assignments


def _ca_trace_secondary_structure_metrics(model: object, binder_chains: list[str]) -> dict[str, Any]:
    assignments: list[str] = []
    for chain_id in binder_chains:
        if str(chain_id) not in model:
            continue
        ca_coords: list[tuple[float, float, float]] = []
        for residue in model[str(chain_id)]:
            if getattr(residue, "id", ("", None, ""))[0] != " " or "CA" not in residue:
                continue
            ca = np.asarray(residue["CA"].coord, dtype=np.float32)
            ca_coords.append((float(ca[0]), float(ca[1]), float(ca[2])))
        if len(ca_coords) >= 4:
            assignments.extend(_ca_trace_assignments(ca_coords))

    if not assignments:
        return {"binder_ca_trace_structure_status": "no binder CA trace"}

    raw_segments = _raw_secondary_structure_segments(assignments)
    structured_segments = _stringent_secondary_structure_segments(assignments)
    helix_like_count = int(sum(value == "H" for value in assignments))
    extended_count = int(sum(value == "E" for value in assignments))
    other_count = int(sum(value not in {"H", "E"} for value in assignments))
    helix_like_elements = int(sum(segment["state"] == "H" for segment in structured_segments))
    extended_elements = int(sum(segment["state"] == "E" for segment in structured_segments))
    total = len(assignments)
    return {
        "binder_ca_trace_structure_status": "ok",
        "binder_ca_trace_structure_method": "ca_trace_angle_i_to_i3_heuristic",
        "binder_ca_trace_c3": "".join(assignments),
        "binder_ca_trace_segments": ";".join(
            f"{segment['start']}-{segment['end']}:{segment['state']}" for segment in structured_segments
        ),
        "binder_ca_trace_raw_segments": ";".join(
            f"{segment['start']}-{segment['end']}:{segment['state']}" for segment in raw_segments
        ),
        "binder_ca_trace_residues": int(total),
        "binder_ca_trace_helix_like_residues": helix_like_count,
        "binder_ca_trace_extended_residues": extended_count,
        "binder_ca_trace_other_residues": other_count,
        "binder_ca_trace_helix_like_elements": helix_like_elements,
        "binder_ca_trace_extended_elements": extended_elements,
        "binder_ca_trace_elements": helix_like_elements + extended_elements,
        "binder_ca_trace_segment_count": int(len(structured_segments)),
        "binder_ca_trace_raw_segment_count": int(len(raw_segments)),
        "binder_ca_trace_helix_like_fraction": helix_like_count / total if total else None,
        "binder_ca_trace_extended_fraction": extended_count / total if total else None,
        "binder_ca_trace_other_fraction": other_count / total if total else None,
    }


def _candidate_binder_secondary_structure_metrics(structure_path: Path, binder_chains: list[str]) -> dict[str, Any]:
    if not binder_chains:
        return {}
    try:
        import pydssp
    except Exception as exc:
        return {"binder_secondary_structure_status": f"pydssp unavailable: {type(exc).__name__}: {exc}"}

    try:
        model = _first_model(_parse_structure(structure_path))
    except Exception as exc:
        return {"binder_secondary_structure_status": f"structure unreadable: {type(exc).__name__}: {exc}"}
    if model is None:
        return {"binder_secondary_structure_status": "no model"}
    ca_trace_metrics = _ca_trace_secondary_structure_metrics(model, binder_chains)

    assignments: list[str] = []
    reconstructed = False
    for chain_id in binder_chains:
        chain_key = str(chain_id)
        if chain_key not in model:
            continue
        coords: list[list[np.ndarray]] = []
        ca_coords: list[tuple[float, float, float]] = []
        for residue in model[chain_key]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] != " ":
                continue
            if "CA" in residue:
                ca = np.asarray(residue["CA"].coord, dtype=np.float32)
                ca_coords.append((float(ca[0]), float(ca[1]), float(ca[2])))
            backbone_atoms: list[np.ndarray] = []
            for atom_name in ("N", "CA", "C", "O"):
                if atom_name not in residue:
                    backbone_atoms = []
                    break
                backbone_atoms.append(np.asarray(residue[atom_name].coord, dtype=np.float32))
            if backbone_atoms:
                coords.append(backbone_atoms)
        assignment_coords = coords
        if len(assignment_coords) < 4 and len(ca_coords) >= 4:
            assignment_coords = _backbone_coords_from_ca_trace(ca_coords)
            reconstructed = True
        try:
            chain_assignments = _pydssp_chain_assignments(pydssp, assignment_coords)
        except Exception:
            continue
        assignments.extend(chain_assignments)

    if not assignments:
        return {"binder_secondary_structure_status": "no complete binder backbone"}

    helix_count = int(sum(value == "H" for value in assignments))
    sheet_count = int(sum(value == "E" for value in assignments))
    coil_count = int(sum(value not in {"H", "E"} for value in assignments))
    raw_segments = _raw_secondary_structure_segments(assignments)
    structured_segments = _stringent_secondary_structure_segments(assignments)
    helix_segment_count = int(sum(segment["state"] == "H" for segment in structured_segments))
    sheet_segment_count = int(sum(segment["state"] == "E" for segment in structured_segments))
    structured_segment_count = helix_segment_count + sheet_segment_count
    structured_count = helix_count + sheet_count
    total = len(assignments)
    return {
        "binder_secondary_structure_status": "ok",
        "binder_secondary_structure_method": "pydssp_c3_ca_rebuilt_backbone" if reconstructed else "pydssp_c3",
        "binder_secondary_structure_reconstructed_backbone": bool(reconstructed),
        "binder_secondary_structure_c3": "".join(value if value in {"H", "E"} else "-" for value in assignments),
        "binder_secondary_structure_segments": ";".join(
            f"{segment['start']}-{segment['end']}:{segment['state']}" for segment in structured_segments
        ),
        "binder_secondary_structure_raw_segments": ";".join(
            f"{segment['start']}-{segment['end']}:{segment['state']}" for segment in raw_segments
        ),
        "binder_secondary_structure_min_helix_length": MIN_HELIX_ELEMENT_LENGTH,
        "binder_secondary_structure_min_sheet_length": MIN_SHEET_ELEMENT_LENGTH,
        "binder_secondary_structure_min_coil_separator": MIN_COIL_SEPARATOR_LENGTH,
        "binder_secondary_structure_residues": int(total),
        "binder_helix_residues": helix_count,
        "binder_sheet_residues": sheet_count,
        "binder_coil_residues": coil_count,
        "binder_structured_residues": structured_count,
        "binder_secondary_structure_elements": int(structured_segment_count),
        "binder_secondary_structure_segment_count": int(len(structured_segments)),
        "binder_secondary_structure_raw_segment_count": int(len(raw_segments)),
        "binder_helix_segment_count": helix_segment_count,
        "binder_sheet_segment_count": sheet_segment_count,
        "binder_helix_fraction": helix_count / total if total else None,
        "binder_sheet_fraction": sheet_count / total if total else None,
        "binder_coil_fraction": coil_count / total if total else None,
        "binder_structured_fraction": structured_count / total if total else None,
        **ca_trace_metrics,
    }


def _with_binder_secondary_structure_metrics(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
    if not binder_chains:
        return candidate
    structure_path = _resolve_candidate_structure_path(run_dir, candidate)
    if structure_path is None:
        return candidate
    metrics = dict(candidate.get("metrics") or {})
    secondary_metrics = _candidate_binder_secondary_structure_metrics(structure_path, binder_chains)
    raw_metadata = dict(candidate.get("raw_metadata") or {})
    if secondary_metrics.get("binder_secondary_structure_reconstructed_backbone") is True:
        reconstructed_path, reconstruction_status = _write_poly_gly_backbone_reconstruction(run_dir, candidate, structure_path)
        if reconstructed_path is not None:
            reconstructed_metrics = _candidate_binder_secondary_structure_metrics(reconstructed_path, binder_chains)
            if reconstructed_metrics.get("binder_secondary_structure_status") == "ok":
                secondary_metrics.update(reconstructed_metrics)
                secondary_metrics["binder_secondary_structure_method"] = "pydssp_c3_pdbfixer_poly_gly_backbone"
                secondary_metrics["binder_secondary_structure_reconstructed_backbone"] = True
            rel_reconstructed = _rel_path(run_dir, reconstructed_path)
            secondary_metrics["binder_secondary_structure_reconstructed_pdb"] = rel_reconstructed
            raw_metadata["poly_gly_backbone_reconstruction_pdb"] = rel_reconstructed
        secondary_metrics["binder_secondary_structure_reconstruction_status"] = reconstruction_status
        raw_metadata["poly_gly_backbone_reconstruction_status"] = reconstruction_status
        raw_metadata["poly_gly_backbone_reconstruction_source_pdb"] = _rel_path(run_dir, structure_path)
    metrics.update(secondary_metrics)
    return {**candidate, "metrics": metrics, "raw_metadata": raw_metadata}


def _with_generation_structure_metrics(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    structure_path = _resolve_candidate_structure_path(run_dir, candidate)
    if structure_path is None:
        return candidate
    metrics = dict(candidate.get("metrics") or {})
    binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
    metrics.update(_candidate_binder_geometry_metrics(structure_path, binder_chains))

    reference_target_path = _resolve_candidate_path(run_dir, candidate.get("target_pdb"), candidate)
    target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    reference_target_chains = [
        str(chain) for chain in (raw_metadata.get("input_target_chains") or target_chains) if str(chain)
    ]
    if reference_target_path is not None and target_chains:
        metrics.update(
            _generation_target_alignment_metrics(
                structure_path,
                reference_target_path,
                target_chains,
                reference_target_chains,
            )
        )
    elif target_chains:
        metrics.setdefault("generation_target_alignment_status", "missing reference target")
    return {**candidate, "metrics": metrics}


def _with_hotspot_ca_contact_metrics(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    hotspots = candidate.get("hotspots") or []
    binder_chains = candidate.get("binder_chains") or []
    target_chains = candidate.get("target_chains") or []
    if not hotspots or not binder_chains or not target_chains:
        return candidate
    structure_path = _resolve_candidate_structure_path(run_dir, candidate)
    if structure_path is None:
        return candidate
    reference_target_path = _resolve_candidate_path(run_dir, candidate.get("target_pdb"), candidate)
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    hotspot_chains = []
    for token in hotspots:
        chain = str(token or "")[:1]
        if chain and chain not in hotspot_chains:
            hotspot_chains.append(chain)
    reference_target_chains = raw_metadata.get("input_target_chains") or hotspot_chains or target_chains
    metrics = dict(candidate.get("metrics") or {})
    try:
        if reference_target_path is not None:
            metrics.update(
                calculate_reference_aligned_hotspot_atom_contact_metrics(
                    structure_path,
                    reference_target_path,
                    hotspots,
                    binder_chains,
                    target_chains,
                    reference_target_chains,
                    contact_cutoff=float(metrics.get("hotspot_atom_contact_cutoff") or 5.0),
                )
            )
        else:
            metrics.update(
                calculate_hotspot_atom_contact_metrics(
                    structure_path,
                    hotspots,
                    binder_chains,
                    target_chains,
                    contact_cutoff=float(metrics.get("hotspot_atom_contact_cutoff") or 5.0),
                )
            )
        metrics.update(
            calculate_hotspot_ca_contact_metrics(
                structure_path,
                hotspots,
                binder_chains,
                target_chains,
                contact_cutoff=float(metrics.get("hotspot_ca_contact_cutoff") or 8.0),
            )
        )
    except Exception as exc:
        metrics["hotspot_ca_contact_error"] = f"{type(exc).__name__}: {exc}"
    return {**candidate, "metrics": metrics}


def write_candidates(run_dir: Path, source_tool: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        normalize_candidate(
            _with_binder_secondary_structure_metrics(
                run_dir,
                _with_generation_structure_metrics(
                    run_dir,
                    _with_hotspot_ca_contact_metrics(run_dir, candidate),
                ),
            )
        )
        for candidate in candidates
    ]
    out_dir = candidates_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    persisted = []
    for candidate in normalized:
        row = dict(candidate)
        for key in ("target_pdb", "complex_pdb", "binder_pdb"):
            if row.get(key):
                row[key] = portable_path(str(row[key]), run_dir=run_dir)
        persisted.append(store_managed_paths(row, run_dir=run_dir))
    candidates_jsonl_path(run_dir).write_text(
        "".join(json.dumps(candidate, sort_keys=True) + "\n" for candidate in persisted)
    )
    write_json(campaign_result_path(run_dir), {"schema_version": CANDIDATE_SCHEMA_VERSION, "source_tool": source_tool, "candidates": persisted})
    return normalized


def read_candidates(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    run_dir = path if path.is_dir() else None
    if path.is_dir():
        path = candidates_jsonl_path(path)
    elif path.parent.name == "normalized_candidates" and path.parent.parent.name == "artifacts":
        run_dir = path.parent.parent.parent
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            candidate = normalize_candidate(json.loads(line))
            candidate = resolve_managed_paths(candidate, run_dir=run_dir)
            for key in ("target_pdb", "complex_pdb", "binder_pdb"):
                value = candidate.get(key)
                if is_portable_path(value) or (value and Path(str(value)).expanduser().is_absolute()):
                    resolved = resolve_stored_path(value, run_dir=run_dir)
                    if resolved is not None:
                        candidate[key] = str(resolved)
            rows.append(candidate)
        except json.JSONDecodeError:
            continue
    return rows


def candidate_stage_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        stage = normalize_stage(candidate.get("stage"))
        counts[stage] = counts.get(stage, 0) + 1
    return counts
