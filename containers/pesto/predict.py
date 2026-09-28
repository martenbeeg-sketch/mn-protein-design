from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch as pt

from src.data_encoding import encode_features, encode_structure, extract_topology
from src.structure import (
    clean_structure,
    concatenate_chains,
    filter_non_atomic_subunits,
    remove_duplicate_tagged_subunits,
    split_by_chain,
    tag_hetatm_chains,
)


MODEL_CODE_DIR = Path("/opt/PeSTo/model/i_v4_1")
INTERFACE_NAMES = ["protein", "dna_rna", "ion", "ligand", "lipid"]
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _load_model(device: pt.device, checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PeSTo checkpoint is missing: {checkpoint}")
    sys.path.insert(0, str(MODEL_CODE_DIR))
    from config import config_model  # type: ignore[import-not-found]
    from model import Model  # type: ignore[import-not-found]

    model = Model(config_model)
    state = pt.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device)


def _input_residues(pdb_text: str) -> list[dict[str, str]]:
    residues: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        residue = line[22:26].strip()
        insertion_code = line[26].strip()
        key = (chain, residue, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        residue_name = line[17:20].strip()
        residues.append(
            {
                "chain": chain,
                "residue": f"{residue}{insertion_code}",
                "amino_acid": AA3_TO_1.get(residue_name, "X"),
                "residue_name": residue_name,
            }
        )
    return residues


def _read_pdb(pdb_text: str) -> dict[str, np.ndarray]:
    data: dict[str, list] = {
        "xyz": [], "name": [], "element": [], "resname": [], "resid": [],
        "het_flag": [], "chain_name": [], "icode": [],
    }
    altloc_seen: set[tuple[str, str, str, str]] = set()
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"} or len(line) < 54:
            continue
        chain = line[21].strip() or "_"
        residue = line[22:26].strip()
        insertion_code = line[26].strip()
        atom_name = line[12:16].strip()
        altloc = line[16].strip()
        altloc_key = (chain, residue, insertion_code, atom_name)
        if altloc and altloc_key in altloc_seen:
            continue
        altloc_seen.add(altloc_key)
        try:
            coordinate = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            residue_id = int(residue)
        except ValueError:
            continue
        element = line[76:78].strip() if len(line) >= 78 else ""
        if not element:
            element = "".join(char for char in atom_name if char.isalpha())[:1].upper() or "X"
        data["xyz"].append(coordinate)
        data["name"].append(atom_name)
        data["element"].append(element.title())
        data["resname"].append(line[17:20].strip())
        data["resid"].append(residue_id)
        data["het_flag"].append("A" if record == "ATOM" else "H")
        data["chain_name"].append(f"{chain}:0")
        data["icode"].append(insertion_code)
    if not data["xyz"]:
        raise RuntimeError("The input PDB does not contain readable coordinates")
    return {
        "xyz": np.asarray(data["xyz"], dtype=np.float32),
        "name": np.asarray(data["name"]),
        "element": np.asarray(data["element"]),
        "resname": np.asarray(data["resname"]),
        "resid": np.asarray(data["resid"], dtype=np.int32),
        "het_flag": np.asarray(data["het_flag"]),
        "chain_name": np.asarray(data["chain_name"]),
        "icode": np.asarray(data["icode"]),
    }


def _preprocess_pdb(pdb_text: str) -> dict[str, dict[str, np.ndarray]]:
    structure = tag_hetatm_chains(clean_structure(_read_pdb(pdb_text)))
    subunits = filter_non_atomic_subunits(split_by_chain(structure))
    return remove_duplicate_tagged_subunits(subunits)


def _collate_features(
    x: pt.Tensor, neighbors: pt.Tensor, features: pt.Tensor, mask: pt.Tensor
) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    packed_neighbors = pt.zeros((x.shape[0], 64), dtype=pt.long)
    packed_neighbors[:, : neighbors.shape[1]] = neighbors + 1
    return x, packed_neighbors, features, mask.float()


def _residue_rows(pdb_text: str, scores: np.ndarray) -> list[dict[str, object]]:
    input_residues = _input_residues(pdb_text)
    if len(input_residues) != len(scores):
        raise RuntimeError(f"PeSTo returned {len(scores)} scores for {len(input_residues)} input residues")
    rows: list[dict[str, object]] = []
    for residue, score in zip(input_residues, scores):
        rows.append({**residue, "score": float(score)})
    return rows


def _scored_pdb(pdb_text: str, rows: list[dict[str, object]]) -> str:
    scores = {(str(row["chain"]), str(row["residue"])): float(row["score"]) for row in rows}
    output: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
            chain = line[21].strip() or "_"
            residue = f"{line[22:26].strip()}{line[26].strip()}"
            score = max(0.0, min(1.0, scores.get((chain, residue), 0.0)))
            padded = line.ljust(66)
            line = f"{padded[:60]}{score:6.2f}{padded[66:]}"
        output.append(line)
    return "\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PeSTo residue-level interface prediction")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interface", choices=INTERFACE_NAMES, default="protein")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("/models/i_v4_1/model_ckpt.pt"))
    args = parser.parse_args()

    use_cuda = args.device == "cuda" or (args.device == "auto" and pt.cuda.is_available())
    if args.device == "cuda" and not pt.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available inside the PeSTo container")
    device = pt.device("cuda" if use_cuda else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdb_text = args.input.read_text(errors="ignore")
    subunits = _preprocess_pdb(pdb_text)
    if not subunits:
        raise RuntimeError("PeSTo preprocessing did not find any usable protein structure")
    structure = concatenate_chains(subunits)
    x, mask = encode_structure(structure)
    features = encode_features(structure)[0]
    neighbors, _, _, _, _ = extract_topology(x, 64)
    x, neighbors, features, mask = _collate_features(x, neighbors, features, mask)

    model = _load_model(device, args.checkpoint)
    with pt.inference_mode():
        logits = model(x.to(device), neighbors.to(device), features.to(device), mask.float().to(device))
        scores = pt.sigmoid(logits[:, INTERFACE_NAMES.index(args.interface)]).cpu().numpy()

    rows = _residue_rows(pdb_text, scores)
    csv_path = args.output_dir / "pesto_residue_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "pesto_scored.pdb").write_text(_scored_pdb(pdb_text, rows))
    print(f"PeSTo {args.interface} prediction: {len(rows)} residues on {device}")
    print(csv_path)


if __name__ == "__main__":
    main()
