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
                "resname": line[17:20].strip().upper(),
                "coord": coord,
            }
        )
    return atoms


def _pdb_ca_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for atom in _pdb_atoms(path):
        if atom["atom"] == "CA":
            atoms.append(atom)
    return atoms


def _cif_ca_atoms_from_text(text: str) -> list[dict[str, Any]]:
    atom_headers: list[str] = []
    in_atom_loop = False
    atoms: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "loop_":
            atom_headers = []
            in_atom_loop = False
            continue
        if line.startswith("_atom_site."):
            atom_headers.append(line.split(".", 1)[1])
            in_atom_loop = True
            continue
        if not in_atom_loop or not line.startswith(("ATOM ", "HETATM ")):
            continue
        parts = line.split()
        if len(parts) < len(atom_headers):
            continue
        row = dict(zip(atom_headers, parts))
        atom_name = (row.get("auth_atom_id") or row.get("label_atom_id") or "").strip("\"'")
        if atom_name != "CA":
            continue
        try:
            coord = np.array(
                [
                    float(row.get("Cartn_x") or row.get("pdbx_model_Cartn_x")),
                    float(row.get("Cartn_y") or row.get("pdbx_model_Cartn_y")),
                    float(row.get("Cartn_z") or row.get("pdbx_model_Cartn_z")),
                ],
                dtype=float,
            )
            resseq = int(float((row.get("auth_seq_id") or row.get("label_seq_id") or "0").strip("\"'")))
        except (TypeError, ValueError):
            continue
        atoms.append(
            {
                "chain": (row.get("auth_asym_id") or row.get("label_asym_id") or "_").strip("\"'"),
                "resseq": resseq,
                "atom": "CA",
                "coord": coord,
            }
        )
    return atoms


def _cif_atoms_from_text(text: str) -> list[dict[str, Any]]:
    atom_headers: list[str] = []
    in_atom_loop = False
    atoms: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "loop_":
            atom_headers = []
            in_atom_loop = False
            continue
        if line.startswith("_atom_site."):
            atom_headers.append(line.split(".", 1)[1])
            in_atom_loop = True
            continue
        if not in_atom_loop:
            continue
        if line.startswith("_"):
            in_atom_loop = False
            continue
        if not line.startswith(("ATOM ", "HETATM ")):
            continue
        parts = line.split()
        if len(parts) < len(atom_headers):
            continue
        row = dict(zip(atom_headers, parts))
        atom_name = (row.get("auth_atom_id") or row.get("label_atom_id") or "").strip("\"'")
        element = (row.get("type_symbol") or atom_name[:1] or "").strip("\"'").upper()
        if element == "H":
            continue
        try:
            coord = np.array(
                [
                    float(row.get("Cartn_x") or row.get("pdbx_model_Cartn_x")),
                    float(row.get("Cartn_y") or row.get("pdbx_model_Cartn_y")),
                    float(row.get("Cartn_z") or row.get("pdbx_model_Cartn_z")),
                ],
                dtype=float,
            )
            resseq = int(float((row.get("auth_seq_id") or row.get("label_seq_id") or "0").strip("\"'")))
        except (TypeError, ValueError):
            continue
        atoms.append(
            {
                "chain": (row.get("auth_asym_id") or row.get("label_asym_id") or "_").strip("\"'"),
                "resseq": resseq,
                "atom": atom_name,
                "resname": (row.get("auth_comp_id") or row.get("label_comp_id") or "UNK").strip("\"'").upper(),
                "coord": coord,
            }
        )
    return atoms


def _structure_atoms(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".cif", ".mmcif"} or path.name.endswith((".cif.gz", ".mmcif.gz")):
        if path.name.endswith(".gz"):
            import gzip

            return _cif_atoms_from_text(gzip.open(path, "rt", errors="ignore").read())
        return _cif_atoms_from_text(path.read_text(errors="ignore"))
    return _pdb_atoms(path)


def _structure_ca_atoms(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".cif", ".mmcif"} or path.name.endswith((".cif.gz", ".mmcif.gz")):
        if path.name.endswith(".gz"):
            import gzip

            return _cif_ca_atoms_from_text(gzip.open(path, "rt", errors="ignore").read())
        return _cif_ca_atoms_from_text(path.read_text(errors="ignore"))
    return _pdb_ca_atoms(path)


_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


def _chain_matches(atom_chain: str, requested_chains: set[str]) -> bool:
    if atom_chain in requested_chains:
        return True
    return any(atom_chain.startswith(chain) or chain.startswith(atom_chain) for chain in requested_chains if chain and atom_chain)


def _hotspot_target_atoms(
    atoms: list[dict[str, Any]],
    hotspot_chain: str,
    hotspot_residue: int,
    target_chain_ids: list[str],
) -> tuple[list[np.ndarray], bool]:
    target_chain_set = set(target_chain_ids)
    all_chains = sorted({str(atom["chain"]) for atom in atoms})
    search_chains = [hotspot_chain]
    search_chains.extend(chain for chain in (target_chain_ids or all_chains) if chain not in search_chains)
    residue_atoms = [
        atom
        for atom in atoms
        if int(atom["resseq"]) == int(hotspot_residue)
        and any(_chain_matches(str(atom["chain"]), {chain}) for chain in search_chains)
        and (not target_chain_set or _chain_matches(str(atom["chain"]), target_chain_set))
    ]
    if not residue_atoms and not target_chain_set:
        residue_atoms = [atom for atom in atoms if int(atom["resseq"]) == int(hotspot_residue)]
    sidechain = [atom["coord"] for atom in residue_atoms if str(atom["atom"]).upper() not in _BACKBONE_ATOMS]
    if sidechain:
        return sidechain, True
    return [atom["coord"] for atom in residue_atoms], False


def calculate_hotspot_atom_contact_metrics(
    structure_path: Path,
    hotspots: list[str] | str | None,
    binder_chains: list[str] | str | None,
    target_chains: list[str] | str | None,
    contact_cutoff: float = 5.0,
) -> dict[str, Any]:
    hotspot_ids = parse_hotspots(hotspots)
    if isinstance(binder_chains, str):
        binder_chain_ids = [chain.strip() for chain in re.split(r"[\s,;]+", binder_chains) if chain.strip()]
    else:
        binder_chain_ids = [str(chain).strip() for chain in (binder_chains or []) if str(chain).strip()]
    if isinstance(target_chains, str):
        target_chain_ids = [chain.strip() for chain in re.split(r"[\s,;]+", target_chains) if chain.strip()]
    else:
        target_chain_ids = [str(chain).strip() for chain in (target_chains or []) if str(chain).strip()]
    metrics: dict[str, Any] = {
        "hotspot_atom_count": len(hotspot_ids),
        "hotspot_atom_found_count": 0,
        "hotspot_atom_contacted_count": 0,
        "hotspot_atom_contact_fraction": None,
        "hotspot_atom_found_contact_fraction": None,
        "hotspot_atom_contacted": "",
        "hotspot_atom_missing": "",
        "hotspot_atom_min_distance": None,
        "hotspot_atom_mean_min_distance": None,
        "hotspot_atom_contact_cutoff": contact_cutoff,
        "binder_atom_count_for_hotspot_contacts": 0,
        "hotspot_contact_mode": "target side-chain heavy atoms vs binder heavy atoms",
    }
    if not hotspot_ids:
        return metrics

    atoms = _structure_atoms(Path(structure_path))
    binder_chain_set = set(binder_chain_ids)
    binder_coords = [
        atom["coord"]
        for atom in atoms
        if _chain_matches(str(atom["chain"]), binder_chain_set)
    ]
    metrics["binder_atom_count_for_hotspot_contacts"] = len(binder_coords)
    if not atoms or not binder_coords:
        metrics["hotspot_atom_contact_error"] = "structure atoms or binder heavy atoms missing"
        return metrics

    distances: list[float] = []
    contacted: list[str] = []
    missing: list[str] = []
    backbone_fallback: list[str] = []
    for hotspot_chain, hotspot_residue in hotspot_ids:
        token = f"{hotspot_chain}{hotspot_residue}"
        hotspot_coords, used_sidechain = _hotspot_target_atoms(atoms, hotspot_chain, hotspot_residue, target_chain_ids)
        if not hotspot_coords:
            missing.append(token)
            continue
        if not used_sidechain:
            backbone_fallback.append(token)
        distance = _min_distance(binder_coords, hotspot_coords)
        if distance is None:
            missing.append(token)
            continue
        distances.append(distance)
        if distance <= float(contact_cutoff):
            contacted.append(token)

    found_count = len(distances)
    metrics["hotspot_atom_found_count"] = found_count
    metrics["hotspot_atom_contacted_count"] = len(contacted)
    metrics["hotspot_atom_contact_fraction"] = len(contacted) / len(hotspot_ids) if hotspot_ids else None
    metrics["hotspot_atom_found_contact_fraction"] = len(contacted) / found_count if found_count else None
    metrics["hotspot_atom_contacted"] = ",".join(contacted)
    metrics["hotspot_atom_missing"] = ",".join(missing)
    metrics["hotspot_atom_min_distance"] = min(distances) if distances else None
    metrics["hotspot_atom_mean_min_distance"] = sum(distances) / len(distances) if distances else None
    if missing:
        metrics["hotspot_atom_contact_warning"] = "some hotspot atoms were not found in the generated target chains"
    if backbone_fallback:
        metrics["hotspot_atom_backbone_fallback"] = ",".join(backbone_fallback)
    return metrics


def _ordered_ca_coords(atoms: list[dict[str, Any]], chain_ids: list[str]) -> list[np.ndarray]:
    chain_set = set(chain_ids)
    coords: list[np.ndarray] = []
    for chain_id in chain_ids:
        chain_atoms = [
            atom
            for atom in atoms
            if str(atom["atom"]) == "CA" and _chain_matches(str(atom["chain"]), {chain_id})
        ]
        chain_atoms.sort(key=lambda atom: int(atom["resseq"]))
        coords.extend(atom["coord"] for atom in chain_atoms)
    if not coords and chain_set:
        chain_atoms = [
            atom
            for atom in atoms
            if str(atom["atom"]) == "CA" and _chain_matches(str(atom["chain"]), chain_set)
        ]
        chain_atoms.sort(key=lambda atom: (str(atom["chain"]), int(atom["resseq"])))
        coords.extend(atom["coord"] for atom in chain_atoms)
    return coords


def _superpose_transform(fixed_coords: list[np.ndarray], moving_coords: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, int] | None:
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
    rotations: list[np.ndarray] = []
    for rotation in (vt_matrix.T @ u_matrix.T, u_matrix @ vt_matrix):
        if np.linalg.det(rotation) < 0:
            adjusted_vt = vt_matrix.copy()
            adjusted_vt[-1, :] *= -1
            rotation = adjusted_vt.T @ u_matrix.T
        rotations.append(rotation)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for rotation in rotations:
        translation = fixed_center - moving_center @ rotation
        transformed = moving @ rotation + translation
        rmsd = float(np.sqrt(np.mean(np.sum((transformed - fixed) ** 2, axis=1))))
        if best is None or rmsd < best[2]:
            best = (rotation, translation, rmsd)
    if best is None:
        return None
    rotation, translation, rmsd = best
    return rotation, translation, rmsd, atom_count


def calculate_reference_aligned_hotspot_atom_contact_metrics(
    structure_path: Path,
    reference_target_path: Path,
    hotspots: list[str] | str | None,
    binder_chains: list[str] | str | None,
    target_chains: list[str] | str | None,
    reference_target_chains: list[str] | str | None,
    contact_cutoff: float = 5.0,
) -> dict[str, Any]:
    metrics = calculate_hotspot_atom_contact_metrics(
        structure_path,
        hotspots,
        binder_chains,
        target_chains,
        contact_cutoff=contact_cutoff,
    )
    metrics["hotspot_contact_frame"] = "generated target residue numbering"
    hotspot_ids = parse_hotspots(hotspots)
    if not hotspot_ids:
        return metrics
    if isinstance(binder_chains, str):
        binder_chain_ids = [chain.strip() for chain in re.split(r"[\s,;]+", binder_chains) if chain.strip()]
    else:
        binder_chain_ids = [str(chain).strip() for chain in (binder_chains or []) if str(chain).strip()]
    if isinstance(target_chains, str):
        target_chain_ids = [chain.strip() for chain in re.split(r"[\s,;]+", target_chains) if chain.strip()]
    else:
        target_chain_ids = [str(chain).strip() for chain in (target_chains or []) if str(chain).strip()]
    if isinstance(reference_target_chains, str):
        reference_chain_ids = [chain.strip() for chain in re.split(r"[\s,;]+", reference_target_chains) if chain.strip()]
    else:
        reference_chain_ids = [str(chain).strip() for chain in (reference_target_chains or []) if str(chain).strip()]
    if not reference_chain_ids:
        reference_chain_ids = [chain for chain, _residue in hotspot_ids]

    reference_atoms = _structure_atoms(Path(reference_target_path))
    moving_atoms = _structure_atoms(Path(structure_path))
    transform = _superpose_transform(
        _ordered_ca_coords(reference_atoms, reference_chain_ids),
        _ordered_ca_coords(moving_atoms, target_chain_ids),
    )
    if transform is None:
        metrics["hotspot_atom_contact_warning"] = "reference target alignment failed; generated residue numbering was used"
        return metrics
    rotation, translation, rmsd, atom_count = transform
    binder_chain_set = set(binder_chain_ids)
    binder_coords = [
        atom["coord"] @ rotation + translation
        for atom in moving_atoms
        if _chain_matches(str(atom["chain"]), binder_chain_set)
    ]
    metrics["binder_atom_count_for_hotspot_contacts"] = len(binder_coords)
    metrics["hotspot_target_alignment_rmsd"] = rmsd
    metrics["hotspot_target_alignment_ca_count"] = atom_count
    metrics["hotspot_contact_frame"] = "reference target after target-CA alignment"
    metrics["hotspot_contact_mode"] = "reference target side-chain heavy atoms vs aligned binder heavy atoms"
    if not binder_coords:
        metrics["hotspot_atom_contact_error"] = "aligned binder heavy atoms missing"
        return metrics

    distances: list[float] = []
    contacted: list[str] = []
    missing: list[str] = []
    backbone_fallback: list[str] = []
    for hotspot_chain, hotspot_residue in hotspot_ids:
        token = f"{hotspot_chain}{hotspot_residue}"
        hotspot_coords, used_sidechain = _hotspot_target_atoms(
            reference_atoms,
            hotspot_chain,
            hotspot_residue,
            reference_chain_ids,
        )
        if not hotspot_coords:
            missing.append(token)
            continue
        if not used_sidechain:
            backbone_fallback.append(token)
        distance = _min_distance(binder_coords, hotspot_coords)
        if distance is None:
            missing.append(token)
            continue
        distances.append(distance)
        if distance <= float(contact_cutoff):
            contacted.append(token)

    found_count = len(distances)
    metrics["hotspot_atom_found_count"] = found_count
    metrics["hotspot_atom_contacted_count"] = len(contacted)
    metrics["hotspot_atom_contact_fraction"] = len(contacted) / len(hotspot_ids) if hotspot_ids else None
    metrics["hotspot_atom_found_contact_fraction"] = len(contacted) / found_count if found_count else None
    metrics["hotspot_atom_contacted"] = ",".join(contacted)
    metrics["hotspot_atom_missing"] = ",".join(missing)
    metrics["hotspot_atom_min_distance"] = min(distances) if distances else None
    metrics["hotspot_atom_mean_min_distance"] = sum(distances) / len(distances) if distances else None
    if missing:
        metrics["hotspot_atom_contact_warning"] = "some hotspot atoms were not found in the reference target"
    if backbone_fallback:
        metrics["hotspot_atom_backbone_fallback"] = ",".join(backbone_fallback)
    return metrics


def calculate_hotspot_ca_contact_metrics(
    structure_path: Path,
    hotspots: list[str] | str | None,
    binder_chains: list[str] | str | None,
    target_chains: list[str] | str | None,
    contact_cutoff: float = 8.0,
) -> dict[str, Any]:
    hotspot_ids = parse_hotspots(hotspots)
    if isinstance(binder_chains, str):
        binder_chain_ids = [chain.strip() for chain in re.split(r"[\s,;]+", binder_chains) if chain.strip()]
    else:
        binder_chain_ids = [str(chain).strip() for chain in (binder_chains or []) if str(chain).strip()]
    if isinstance(target_chains, str):
        target_chain_ids = [chain.strip() for chain in re.split(r"[\s,;]+", target_chains) if chain.strip()]
    else:
        target_chain_ids = [str(chain).strip() for chain in (target_chains or []) if str(chain).strip()]
    metrics: dict[str, Any] = {
        "hotspot_ca_count": len(hotspot_ids),
        "hotspot_ca_found_count": 0,
        "hotspot_ca_contacted_count": 0,
        "hotspot_ca_contact_fraction": None,
        "hotspot_ca_found_contact_fraction": None,
        "hotspot_ca_contacted": "",
        "hotspot_ca_missing": "",
        "hotspot_ca_min_distance": None,
        "hotspot_ca_mean_min_distance": None,
        "hotspot_ca_contact_cutoff": contact_cutoff,
        "binder_ca_count_for_hotspot_contacts": 0,
    }
    if not hotspot_ids:
        return metrics

    atoms = _structure_ca_atoms(Path(structure_path))
    binder_coords = [
        atom["coord"]
        for atom in atoms
        if atom["chain"] in set(binder_chain_ids)
    ]
    metrics["binder_ca_count_for_hotspot_contacts"] = len(binder_coords)
    if not atoms or not binder_coords:
        metrics["hotspot_ca_contact_error"] = "structure CA atoms or binder CA atoms missing"
        return metrics

    ca_by_chain_residue: dict[tuple[str, int], np.ndarray] = {
        (str(atom["chain"]), int(atom["resseq"])): atom["coord"] for atom in atoms
    }
    all_chains = sorted({str(atom["chain"]) for atom in atoms})
    target_chain_set = set(target_chain_ids)
    distances: list[float] = []
    contacted: list[str] = []
    missing: list[str] = []

    for hotspot_chain, hotspot_residue in hotspot_ids:
        token = f"{hotspot_chain}{hotspot_residue}"
        coordinate = ca_by_chain_residue.get((hotspot_chain, hotspot_residue))
        if coordinate is None:
            search_chains = target_chain_ids or all_chains
            for chain in search_chains:
                coordinate = ca_by_chain_residue.get((chain, hotspot_residue))
                if coordinate is not None:
                    break
        if coordinate is None and not target_chain_set:
            for chain in all_chains:
                coordinate = ca_by_chain_residue.get((chain, hotspot_residue))
                if coordinate is not None:
                    break
        if coordinate is None:
            missing.append(token)
            continue
        deltas = np.asarray(binder_coords) - coordinate
        distance = float(np.sqrt(np.sum(deltas * deltas, axis=1)).min())
        distances.append(distance)
        if distance <= float(contact_cutoff):
            contacted.append(token)

    found_count = len(distances)
    metrics["hotspot_ca_found_count"] = found_count
    metrics["hotspot_ca_contacted_count"] = len(contacted)
    metrics["hotspot_ca_contact_fraction"] = len(contacted) / len(hotspot_ids) if hotspot_ids else None
    metrics["hotspot_ca_found_contact_fraction"] = len(contacted) / found_count if found_count else None
    metrics["hotspot_ca_contacted"] = ",".join(contacted)
    metrics["hotspot_ca_missing"] = ",".join(missing)
    metrics["hotspot_ca_min_distance"] = min(distances) if distances else None
    metrics["hotspot_ca_mean_min_distance"] = sum(distances) / len(distances) if distances else None
    if missing:
        metrics["hotspot_ca_contact_warning"] = "some hotspot CA atoms were not found in the generated target chains"
    return metrics


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
