from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley

from mn_protein_design.core.jobs import read_json
from mn_protein_design.workflows.detection import detection_jobs_for_target


BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
AA3_TO_1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def sidechain_sasa_by_residue(pdb_path: Path) -> dict[tuple[str, int], float]:
    """Return per-residue side-chain SASA in square angstroms."""
    structure = PDBParser(QUIET=True).get_structure("target", str(pdb_path))
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(structure, level="A")
    values: dict[tuple[str, int], float] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].strip():
                    continue
                sidechain_sasa = sum(
                    float(getattr(atom, "sasa", 0.0) or 0.0)
                    for atom in residue
                    if atom.get_name().strip().upper() not in BACKBONE_ATOMS
                )
                values[(chain.id.strip() or "_", int(residue.id[1]))] = sidechain_sasa
        break
    return values


def _amino_acids_by_residue(pdb_path: Path) -> dict[tuple[str, int], str]:
    values: dict[tuple[str, int], str] = {}
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  ") or len(line) < 26:
            continue
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        values.setdefault((line[21].strip() or "_", residue), AA3_TO_1.get(line[17:20].strip(), "X"))
    return values


def _masif_residue_scores(surface_path: Path, prediction_path: Path, target_pdb: Path) -> pd.DataFrame:
    lines = surface_path.read_text(errors="ignore").splitlines()
    try:
        start = lines.index("end_header") + 1
    except ValueError:
        return pd.DataFrame()
    scores = np.asarray(np.load(prediction_path)).reshape(-1)
    vertices = np.asarray([[float(value) for value in line.split()[:3]] for line in lines[start : start + len(scores)]], dtype=float)
    if len(vertices) != len(scores):
        return pd.DataFrame()
    atoms: list[tuple[str, int, list[float]]] = []
    for line in target_pdb.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        try:
            atoms.append((line[21].strip() or "_", int(line[22:26]), [float(line[30:38]), float(line[38:46]), float(line[46:54])]))
        except ValueError:
            continue
    if not atoms:
        return pd.DataFrame()
    atom_xyz = np.asarray([atom[2] for atom in atoms])
    values: dict[tuple[str, int], float] = {}
    for vertex, score in zip(vertices, scores):
        index = int(np.argmin(np.sum((atom_xyz - vertex) ** 2, axis=1)))
        if float(np.sum((atom_xyz[index] - vertex) ** 2)) > 16.0:
            continue
        key = atoms[index][:2]
        values[key] = max(values.get(key, 0.0), float(score))
    return pd.DataFrame([{"chain": chain, "residue": residue, "amino_acid": "", "score": score} for (chain, residue), score in values.items()])


def detection_score_sources(target_pdb: Path, target_chains: list[str]) -> dict[str, pd.DataFrame]:
    """Load normalized residue scores from completed detection runs for a target."""
    sources: dict[str, pd.DataFrame] = {}
    allowed_chains = set(target_chains)
    amino_acids = _amino_acids_by_residue(target_pdb)
    for job in detection_jobs_for_target(target_pdb):
        if job.get("status") != "completed":
            continue
        run_dir = Path(str(job["run_dir"]))
        result = read_json(run_dir / "result.json")
        downstream = result.get("downstream_artifacts") or {}
        paths = [*(downstream.get("residue_scores") or []), *(downstream.get("hotspot_tables") or [])]
        tool = str(job.get("tool") or "detection")
        if tool == "masif_seed":
            prediction_paths = downstream.get("site_predictions") or []
            surface_paths = downstream.get("predicted_surfaces") or []
            for prediction_text, surface_text in zip(prediction_paths, surface_paths):
                table = _masif_residue_scores(run_dir / surface_text, run_dir / prediction_text, target_pdb)
                if not table.empty:
                    table["amino_acid"] = table.apply(lambda row: amino_acids.get((str(row["chain"]), int(row["residue"])), "X"), axis=1)
                    sources[f"{tool} | {job['job_code']} | projected surface scores"] = table
        for path_text in paths:
            path = run_dir / str(path_text)
            if not path.exists():
                continue
            try:
                raw = pd.read_csv(path)
            except Exception:
                continue
            if {"chain", "residue", "amino_acid", "score"}.issubset(raw.columns):
                table = raw[["chain", "residue", "amino_acid", "score"]].copy()
            elif {"Chain", "Residue Index", "Sequence", "Binding site probability"}.issubset(raw.columns):
                table = pd.DataFrame({
                    "chain": raw["Chain"], "residue": raw["Residue Index"],
                    "amino_acid": raw["Sequence"], "score": raw["Binding site probability"],
                })
            elif {"aa_id", "score"}.issubset(raw.columns) and len(allowed_chains) == 1:
                chain = next(iter(allowed_chains))
                table = pd.DataFrame({"chain": chain, "residue": raw["aa_id"], "amino_acid": "", "score": raw["score"]})
            else:
                continue
            table["chain"] = table["chain"].astype(str)
            table["residue"] = pd.to_numeric(table["residue"], errors="coerce")
            table["score"] = pd.to_numeric(table["score"], errors="coerce")
            table = table.dropna(subset=["residue", "score"])
            table["residue"] = table["residue"].astype(int)
            table = table[table["chain"].isin(allowed_chains)].copy()
            table["amino_acid"] = table.apply(
                lambda row: str(row["amino_acid"]).strip() or amino_acids.get((str(row["chain"]), int(row["residue"])), "X"),
                axis=1,
            )
            if not table.empty:
                sources[f"{tool} | {job['job_code']} | {path.name}"] = table
    return sources
