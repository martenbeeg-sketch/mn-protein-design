from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_hotspots(hotspots: list[str] | str | None) -> list[tuple[str, int]]:
    if not hotspots:
        return []
    if isinstance(hotspots, str):
        tokens = [token.strip() for token in re.split(r"[\s,;]+", hotspots) if token.strip()]
    else:
        tokens = [str(token).strip() for token in hotspots if str(token).strip()]
    parsed: list[tuple[str, int]] = []
    for token in tokens:
        token = token.replace(":", "")
        match = re.fullmatch(r"([A-Za-z])(\d+)", token)
        if match:
            parsed.append((match.group(1).upper(), int(match.group(2))))
    return parsed


def _pdb_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() if len(line) >= 78 else atom_name[:1]).upper()
        if element == "H":
            continue
        try:
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
            resseq = int(line[22:26])
        except ValueError:
            continue
        atoms.append(
            {
                "chain": line[21].strip() or "_",
                "resseq": resseq,
                "atom": atom_name,
                "coord": coord,
            }
        )
    return atoms


def _residue_atoms(atoms: list[dict[str, Any]], chain_id: str) -> dict[int, list[np.ndarray]]:
    residues: dict[int, list[np.ndarray]] = {}
    for atom in atoms:
        if atom["chain"] == chain_id:
            residues.setdefault(int(atom["resseq"]), []).append(atom["coord"])
    return residues


def _ca_coords(atoms: list[dict[str, Any]], chain_id: str) -> list[np.ndarray]:
    return [atom["coord"] for atom in atoms if atom["chain"] == chain_id and atom["atom"] == "CA"]


def _min_distance(coords_a: list[np.ndarray], coords_b: list[np.ndarray]) -> float | None:
    if not coords_a or not coords_b:
        return None
    best = math.inf
    for coord_a in coords_a:
        deltas = np.asarray(coords_b) - coord_a
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        best = min(best, float(distances.min()))
    return best if math.isfinite(best) else None


def _resolve_hotspot_chain(
    hotspot_chain: str,
    hotspot_residue: int,
    target_chain: str,
    target_residues: dict[int, list[np.ndarray]],
    atoms: list[dict[str, Any]],
) -> str:
    if hotspot_residue in target_residues:
        return target_chain
    if any(atom["chain"] == hotspot_chain and atom["resseq"] == hotspot_residue for atom in atoms):
        return hotspot_chain
    return target_chain if target_residues else hotspot_chain


def _target_contig_mapping(contig: str | None, target_residues: dict[int, list[np.ndarray]]) -> tuple[str | None, int | None]:
    if not contig or not target_residues:
        return None, None
    match = re.search(r"([A-Za-z])(\d+)-(\d+)", contig)
    if not match:
        return None, None
    input_chain = match.group(1).upper()
    input_start = int(match.group(2))
    output_start = min(target_residues)
    return input_chain, output_start - input_start


def calculate_hotspot_metrics(
    structure_path: Path,
    hotspots: list[str] | str | None,
    binder_chain: str = "A",
    target_chain: str = "B",
    contact_cutoff: float = 8.0,
    contig: str | None = None,
) -> dict[str, Any]:
    hotspot_ids = parse_hotspots(hotspots)
    atoms = _pdb_atoms(Path(structure_path))
    binder_residues = _residue_atoms(atoms, binder_chain)
    target_residues = _residue_atoms(atoms, target_chain)
    contig_chain, target_offset = _target_contig_mapping(contig, target_residues)
    binder_coords = [coord for coords in binder_residues.values() for coord in coords]
    binder_ca = _ca_coords(atoms, binder_chain)

    metrics: dict[str, Any] = {
        "hotspot_count": len(hotspot_ids),
        "hotspots_contacted": 0,
        "hotspot_contact_fraction": None,
        "hotspot_contact_pairs": 0,
        "min_binder_to_hotspot_distance": None,
        "mean_binder_to_hotspot_distance": None,
        "binder_interface_contacts": 0,
        "hotspot_interface_contact_fraction": None,
        "hotspot_contact_cutoff": contact_cutoff,
        "binder_radius_of_gyration": None,
        "binder_end_to_end_distance": None,
        "binder_ca_count": len(binder_ca),
    }

    if binder_ca:
        ca_array = np.asarray(binder_ca)
        center = ca_array.mean(axis=0)
        metrics["binder_radius_of_gyration"] = float(np.sqrt(np.mean(np.sum((ca_array - center) ** 2, axis=1))))
        metrics["binder_end_to_end_distance"] = float(np.linalg.norm(ca_array[-1] - ca_array[0])) if len(ca_array) > 1 else 0.0

    if not binder_residues or not target_residues:
        metrics["hotspot_metrics_error"] = "binder or target chain missing"
        return metrics

    binder_interface_contacts = 0
    for binder_atoms in binder_residues.values():
        for target_atoms in target_residues.values():
            distance = _min_distance(binder_atoms, target_atoms)
            if distance is not None and distance <= contact_cutoff:
                binder_interface_contacts += 1
    metrics["binder_interface_contacts"] = binder_interface_contacts

    if not hotspot_ids:
        return metrics

    hotspot_distances: list[float] = []
    contacted_hotspots: set[tuple[str, int]] = set()
    hotspot_contact_pairs = 0
    for hotspot_chain, hotspot_residue in hotspot_ids:
        resolved_residue = hotspot_residue
        if target_offset is not None and contig_chain and hotspot_chain == contig_chain:
            resolved_residue = hotspot_residue + target_offset
            resolved_chain = target_chain
        else:
            resolved_chain = _resolve_hotspot_chain(hotspot_chain, hotspot_residue, target_chain, target_residues, atoms)
        residue_atoms = _residue_atoms(atoms, resolved_chain).get(resolved_residue, [])
        distance = _min_distance(binder_coords, residue_atoms)
        if distance is None:
            continue
        hotspot_distances.append(distance)
        if distance <= contact_cutoff:
            contacted_hotspots.add((hotspot_chain, hotspot_residue))
        for binder_atoms in binder_residues.values():
            pair_distance = _min_distance(binder_atoms, residue_atoms)
            if pair_distance is not None and pair_distance <= contact_cutoff:
                hotspot_contact_pairs += 1

    hotspot_count = len(hotspot_ids)
    metrics["hotspots_contacted"] = len(contacted_hotspots)
    metrics["hotspot_contact_fraction"] = len(contacted_hotspots) / hotspot_count if hotspot_count else None
    metrics["hotspot_contact_pairs"] = hotspot_contact_pairs
    metrics["min_binder_to_hotspot_distance"] = min(hotspot_distances) if hotspot_distances else None
    metrics["mean_binder_to_hotspot_distance"] = (
        sum(hotspot_distances) / len(hotspot_distances) if hotspot_distances else None
    )
    metrics["hotspot_interface_contact_fraction"] = (
        hotspot_contact_pairs / binder_interface_contacts if binder_interface_contacts else None
    )
    if not hotspot_distances:
        metrics["hotspot_metrics_error"] = "hotspot residues were not found in the target chain"
    return metrics


def passes_hotspot_prefilter(
    metrics: dict[str, Any],
    min_contact_fraction: float = 0.25,
    max_min_distance: float = 10.0,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    hotspot_count = int(metrics.get("hotspot_count") or 0)
    if hotspot_count == 0:
        return True, failures
    contact_fraction = metrics.get("hotspot_contact_fraction")
    min_distance = metrics.get("min_binder_to_hotspot_distance")
    if contact_fraction is not None and float(contact_fraction) < float(min_contact_fraction):
        failures.append(f"hotspot_contact_fraction < {min_contact_fraction:g}")
    if min_distance is not None and float(min_distance) > float(max_min_distance):
        failures.append(f"min_binder_to_hotspot_distance > {max_min_distance:g}")
    if contact_fraction is None and min_distance is None:
        failures.append("hotspot metrics unavailable")
    return not failures, failures
