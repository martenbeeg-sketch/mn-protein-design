#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time

import jax
import numpy as np
from colabdesign import mk_af_model
from colabdesign.af.alphafold.common import protein


METRICS = {
    "rmsd": "target_aligned_binder_rmsd",
    "plddt": "binder_plddt",
    "pae": "binder_pae",
    "ptm": "ptm",
    "con": "con_loss",
    "i_pae": "ipae",
    "i_ptm": "iptm",
    "i_con": "i_con_loss",
}


def add_cyclic_offset(model, offset_type: int = 2):
    def cyclic_offset(length: int):
        i = np.arange(length)
        ij = np.stack([i, i + length], -1)
        offset = i[:, None] - i[None, :]
        c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
        if offset_type >= 2:
            idx = c_offset < np.abs(offset)
            c_offset[idx] = -c_offset[idx]
        if offset_type == 3:
            idx = np.abs(c_offset) > 2
            c_offset[idx] = (32 * c_offset[idx]) / abs(c_offset[idx])
        return c_offset * np.sign(offset)

    idx = model._inputs["residue_index"]
    offset = np.array(idx[:, None] - idx[None, :])
    offset[model._target_len :, model._target_len :] = cyclic_offset(model._binder_len)
    model._inputs["offset"] = offset


def get_pdb_total_length(pdb_path: str) -> int:
    unique_residues = set()
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                unique_residues.add((line[21].strip(), line[22:26].strip(), line[26].strip()))
    return len(unique_residues)


def get_binder_sequence(pdb_path: str, chain_id: str) -> str:
    sidecar = os.path.splitext(pdb_path)[0] + ".binder_sequence.txt"
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as handle:
            return "".join(handle.read().split()).upper()

    residue_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    residues: list[str] = []
    seen: set[tuple[str, str]] = set()
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or line[21].strip() != chain_id:
                continue
            residue_key = (line[22:26].strip(), line[26].strip())
            if residue_key in seen:
                continue
            seen.add(residue_key)
            residues.append(residue_map.get(line[17:20].strip().upper(), "X"))
    return "".join(residues)


def get_target_chains(pdb_path: str, fallback: str) -> str:
    sidecar = os.path.splitext(pdb_path)[0] + ".target_chains.txt"
    if os.path.exists(sidecar):
        with open(sidecar, "r", encoding="utf-8") as handle:
            chains = ",".join(
                part.strip()
                for part in handle.read().split(",")
                if part.strip()
            )
        if chains:
            return chains
    return fallback


def get_binder_source_chains(pdb_path: str, fallback: str = "A") -> list[str]:
    sidecar = os.path.splitext(pdb_path)[0] + ".binder_source_chains.txt"
    if not os.path.exists(sidecar):
        return [fallback]
    with open(sidecar, "r", encoding="utf-8") as handle:
        chains = [part.strip() for part in handle.read().split(",") if part.strip()]
    return chains or [fallback]


def _ca_coordinates(pdb_path: str, chains: list[str] | None = None) -> np.ndarray:
    coordinates = []
    chain_set = set(chains or [])
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if chain_set and line[21].strip() not in chain_set:
                continue
            try:
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
    return np.asarray(coordinates, dtype=float)


def _aligned_ca_rmsd(reference_pdb: str, predicted_pdb: str, reference_chains: list[str]) -> float | None:
    reference = _ca_coordinates(reference_pdb, reference_chains)
    predicted = _ca_coordinates(predicted_pdb)
    count = min(len(reference), len(predicted))
    if count < 3:
        return None
    reference = reference[:count]
    predicted = predicted[:count]
    reference_centered = reference - reference.mean(axis=0)
    predicted_centered = predicted - predicted.mean(axis=0)
    u, _s, vt = np.linalg.svd(predicted_centered.T @ reference_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    aligned = predicted_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1))))


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.array(value)
    except Exception:
        return None


def _extract_pae_matrix(model) -> np.ndarray | None:
    aux = getattr(model, "aux", {}) or {}
    candidates = []
    if isinstance(aux, dict):
        candidates.append(aux.get("pae"))
        outputs = aux.get("outputs", {})
        if isinstance(outputs, dict):
            candidates.append(outputs.get("predicted_aligned_error"))
        all_aux = aux.get("all", {})
        if isinstance(all_aux, dict):
            candidates.append(all_aux.get("pae"))
            candidates.append(all_aux.get("predicted_aligned_error"))

    for candidate in candidates:
        arr = _to_numpy(candidate)
        if arr is None:
            continue
        arr = np.squeeze(arr)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            arr = arr.astype(float)
            if np.nanmax(arr) <= 1.5:
                arr = arr * 31.0
            return arr
    return None


def _flip_target_binder_matrix(matrix: np.ndarray, target_len: int, binder_len: int) -> np.ndarray:
    order = np.concatenate([np.arange(target_len, target_len + binder_len), np.arange(0, target_len)])
    return matrix[np.ix_(order, order)]


def _save_pae_json(model, out_path: str, metrics: dict):
    pae_matrix = _extract_pae_matrix(model)
    if pae_matrix is None:
        return
    pae_matrix = _flip_target_binder_matrix(pae_matrix, model._target_len, model._binder_len)
    payload = {
        "predicted_aligned_error": pae_matrix.tolist(),
        "pae": pae_matrix.tolist(),
        "max_predicted_aligned_error": 31.0,
    }
    for key in ("ptm", "iptm"):
        if metrics.get(key) is not None:
            payload[key] = metrics[key]
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _save_monomer_pae_json(model, out_path: str, metrics: dict):
    pae_matrix = _extract_pae_matrix(model)
    if pae_matrix is None:
        return
    payload = {
        "predicted_aligned_error": pae_matrix.tolist(),
        "pae": pae_matrix.tolist(),
        "max_predicted_aligned_error": 31.0,
    }
    for key in ("ptm", "iptm"):
        if metrics.get(key) is not None:
            payload[key] = metrics[key]
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def save_binder_design_pdb(model, filename: str):
    aux = model._tmp["best"]["aux"] if "aux" in model._tmp.get("best", {}) else model.aux
    aux = aux["all"]
    payload = {key: aux[key] for key in ["aatype", "residue_index", "atom_positions", "atom_mask"]}
    payload["b_factors"] = 100 * payload["atom_mask"] * aux["plddt"][..., None]

    for key in list(payload):
        assert payload[key].shape[1] == model._target_len + model._binder_len
        payload[key] = np.concatenate([payload[key][:, model._target_len :], payload[key][:, : model._target_len]], axis=1)

    def to_pdb_str(x, model_number=None):
        pdb_str = protein.to_pdb(protein.Protein(**x))
        binder_mapping = dict(zip(x["residue_index"][: model._binder_len], range(1, model._binder_len + 1)))
        lines = []
        for line in pdb_str.splitlines()[1:-2]:
            if line.startswith(("ATOM", "HETATM")):
                resno = int(line[22:26].strip())
                if resno in binder_mapping:
                    lines.append(line[:21] + "A" + str(binder_mapping[resno]).rjust(4) + line[26:])
                else:
                    lines.append(line[:21] + "B" + line[22:])
        body = "\n".join(lines)
        if model_number is not None:
            return f"MODEL{model_number:8}\n{body}\nENDMDL\n"
        return body

    pdb_out = ""
    for index in range(payload["atom_positions"].shape[0]):
        pdb_out += to_pdb_str(jax.tree_util.tree_map(lambda x: x[index], payload), index + 1)
    pdb_out += "END\n"
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(pdb_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_name")
    parser.add_argument("--params", required=True)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--model-count", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--use-binder-template", action="store_true", default=False)
    parser.add_argument("--use-interface-template", action="store_true", default=False)
    parser.add_argument(
        "--target-template-only",
        action="store_true",
        default=False,
        help="Use the staged target-only PDB as template input and read the binder sequence from its sidecar file.",
    )
    parser.add_argument(
        "--single-chain-template-only",
        action="store_true",
        default=False,
        help="Fold one staged chain with fixbb/template inputs; used for monomer capacity tests.",
    )
    parser.add_argument("--multimer", action="store_true", default=False)
    parser.add_argument(
        "--binder-multimer",
        action="store_true",
        default=False,
        help="Use AF2-multimer-v3 parameters for the independent binder-alone prediction.",
    )
    parser.add_argument("--designed_chains", default="A")
    parser.add_argument("--cyclic", action="store_true", default=False)
    parser.add_argument("--hotspot", default=None)
    options = parser.parse_args()

    if options.use_interface_template:
        assert options.use_binder_template

    if options.single_chain_template_only:
        options.target_template_only = False

    model_names = [
        f"model_{index}_multimer_v3" if options.multimer else f"model_{index}_ptm"
        for index in range(1, options.model_count + 1)
    ]
    binder_uses_multimer = options.binder_multimer
    binder_model_names = [
        f"model_{index}_multimer_v3" if binder_uses_multimer else f"model_{index}_ptm"
        for index in range(1, options.model_count + 1)
    ]
    if options.single_chain_template_only:
        model = mk_af_model(
            protocol="fixbb",
            data_dir=options.params,
            use_multimer=options.multimer,
            model_names=model_names,
            use_templates=True,
            use_initial_guess=True,
        )
        binder_model = None
    else:
        model = mk_af_model(
            protocol="binder",
            data_dir=options.params,
            use_multimer=options.multimer,
            model_names=model_names,
            use_initial_guess=not options.target_template_only,
        )
        binder_model = mk_af_model(
            protocol="hallucination",
            data_dir=options.params,
            use_multimer=binder_uses_multimer,
            model_names=binder_model_names,
            use_initial_guess=False,
        )
    paths = sorted(glob.glob(os.path.join(options.input_dir, "*.pdb")))
    total_lengths = {path: get_pdb_total_length(path) for path in paths}
    paths = sorted(paths, key=lambda path: total_lengths[path])
    print(f"Processing {len(paths):,} PDBs")

    os.makedirs(options.output_name, exist_ok=True)
    with open(options.output_name.rstrip("/") + ".jsonl", "wt", encoding="utf-8") as handle:
        for index, path in enumerate(paths, start=1):
            basename = os.path.basename(path).removesuffix(".pdb")
            print(f"Predicting PDB {index:,}/{len(paths):,}: {basename}")
            start_time = time.time()
            target_chain = get_target_chains(
                path,
                "B" if options.designed_chains == "A" else "A",
            )
            if options.designed_chains not in {"A", "B"}:
                raise NotImplementedError("Expected binder chain to be A or B")

            binder_sequence = get_binder_sequence(path, options.designed_chains)
            if options.single_chain_template_only:
                if not binder_sequence or "X" in binder_sequence:
                    raise ValueError(f"Could not extract a complete sequence from chain {options.designed_chains} in {path}")
                model.prep_inputs(
                    pdb_filename=path,
                    chain=options.designed_chains,
                    rm_template=False,
                    rm_template_seq=False,
                    rm_template_sc=False,
                    rm_template_ic=False,
                )
            elif options.target_template_only:
                if not binder_sequence or "X" in binder_sequence:
                    raise ValueError(f"Could not extract a complete binder sequence from chain {options.designed_chains} in {path}")
                model.prep_inputs(
                    path,
                    binder_chain=None,
                    binder_len=len(binder_sequence),
                    target_chain=target_chain,
                    rm_target=False,
                    hotspot=options.hotspot if options.hotspot else None,
                )
            else:
                model.prep_inputs(
                    path,
                    binder_chain=options.designed_chains,
                    target_chain=target_chain,
                    rm_target=False,
                    rm_binder=not options.use_binder_template,
                    rm_template_ic=not options.use_interface_template,
                    hotspot=options.hotspot if options.hotspot else None,
                )
            if options.cyclic:
                add_cyclic_offset(model, offset_type=2)
            if options.single_chain_template_only:
                model.set_seq(seq=binder_sequence)
            elif options.target_template_only:
                model.set_seq(seq=binder_sequence)
            else:
                model.set_seq(mode="wildtype")
            suffix = os.path.basename(options.output_name.rstrip("/"))
            out_base = os.path.join(options.output_name, f"{basename}_{suffix}")
            model_rows = []
            model_outputs = []
            model.set_opt(num_recycles=options.num_recycles)
            for model_index in range(options.model_count):
                model.predict(models=[model_index], num_recycles=options.num_recycles, verbose=False)
                if options.single_chain_template_only:
                    log = model.aux["log"]
                    row = {
                        "binder_plddt": float(log.get("plddt")) * 100.0 if log.get("plddt") is not None else None,
                        "binder_pae": float(log.get("pae")) * 31.0 if log.get("pae") is not None else None,
                        "ptm": float(log.get("ptm")) if log.get("ptm") is not None else None,
                        "target_aligned_binder_rmsd": None,
                        "ipae": None,
                        "iptm": None,
                        "con_loss": None,
                        "i_con_loss": None,
                        "interface_target_residues": "",
                    }
                else:
                    row = {new_key: model.aux["log"].get(old_key) for old_key, new_key in METRICS.items()}
                    row["binder_plddt"] *= 100
                    row["binder_pae"] *= 31.0
                    row["ipae"] *= 31.0
                    ca_pos = model.aux["atom_positions"][:, 1]
                    ca_dist = np.sqrt(
                        np.square(
                            ca_pos[model._target_len :, None]
                            - ca_pos[None, : model._target_len]
                        ).sum(axis=-1)
                        + 1e-8
                    )
                    target_interface_res = model.aux["residue_index"][: model._target_len][
                        ca_dist.min(axis=0) <= 8
                    ]
                    row["interface_target_residues"] = ",".join(
                        f"B{pos}" for pos in target_interface_res
                    )
                model_number = model_index + 1
                model_pdb = f"{out_base}_model{model_number}.pdb"
                model_pae = f"{out_base}_model{model_number}_pae.json"
                if options.single_chain_template_only:
                    model.save_pdb(model_pdb)
                    _save_monomer_pae_json(model, model_pae, row)
                else:
                    save_binder_design_pdb(model, model_pdb)
                    _save_pae_json(model, model_pae, row)
                model_rows.append(row)
                model_outputs.append((model_pdb, model_pae))

            metrics = {
                "id": basename,
                "model_count": len(model_rows),
                "af2_validation_parameter_family": (
                    "multimer_v3" if options.multimer else "ptm"
                ),
                "complex_af2_parameter_family": (
                    "multimer_v3" if options.multimer else "ptm"
                ),
                "binder_fold_af2_parameter_family": (
                    "multimer_v3" if binder_uses_multimer else "ptm"
                ),
                "complex_prediction_protocol": (
                    "single_chain_template_fold" if options.single_chain_template_only else "target_binder_complex"
                ),
                "binder_fold_prediction_protocol": (
                    "not_applicable_single_chain" if options.single_chain_template_only else "binder_alone"
                ),
            }
            for model_number, row in enumerate(model_rows, start=1):
                for key, value in row.items():
                    metrics[f"model_{model_number}_{key}"] = value
            for key in METRICS.values():
                values = [float(row[key]) for row in model_rows if row.get(key) is not None]
                metrics[key] = sum(values) / len(values) if values else None
                metrics[f"average_{key}"] = metrics[key]
            metrics["time"] = time.time() - start_time

            best_index = max(
                range(len(model_rows)),
                key=lambda idx: float(model_rows[idx].get("binder_plddt") or 0.0),
            )
            shutil.copy2(model_outputs[best_index][0], f"{out_base}.pdb")
            if os.path.exists(model_outputs[best_index][1]):
                shutil.copy2(model_outputs[best_index][1], f"{out_base}_pae.json")
            metrics["representative_model"] = best_index + 1
            reference_pdb = os.path.join(options.input_dir, "references", f"{basename}.pdb")
            reference_chains = get_binder_source_chains(path)
            metrics["interface_target_residues"] = model_rows[best_index][
                "interface_target_residues"
            ]

            if options.single_chain_template_only:
                metrics["monomer_refolding_rmsd"] = (
                    _aligned_ca_rmsd(reference_pdb, model_outputs[best_index][0], reference_chains)
                    if os.path.exists(reference_pdb)
                    else None
                )
                metrics["average_binder_only_plddt"] = None
                metrics["average_binder_only_pae"] = None
                metrics["average_binder_rmsd"] = metrics["monomer_refolding_rmsd"]
            else:
                binder_model.prep_inputs(length=len(binder_sequence))
                binder_model.set_seq(seq=binder_sequence)
                binder_model.set_opt(num_recycles=options.num_recycles)
                for model_index in range(options.model_count):
                    binder_model.predict(models=[model_index], num_recycles=options.num_recycles, verbose=False)
                    binder_number = model_index + 1
                    binder_pdb = f"{out_base}_binder_model{binder_number}.pdb"
                    binder_model.save_pdb(binder_pdb)
                    binder_log = binder_model.aux["log"]
                    binder_plddt = binder_log.get("plddt")
                    binder_pae = binder_log.get("pae")
                    metrics[f"model_{binder_number}_binder_only_plddt"] = (
                        float(binder_plddt) * 100.0 if binder_plddt is not None else None
                    )
                    metrics[f"model_{binder_number}_binder_only_pae"] = (
                        float(binder_pae) * 31.0 if binder_pae is not None else None
                    )
                    metrics[f"model_{binder_number}_binder_rmsd"] = (
                        _aligned_ca_rmsd(reference_pdb, binder_pdb, reference_chains)
                        if os.path.exists(reference_pdb)
                        else None
                    )
            if not options.single_chain_template_only:
                for metric_name in ("binder_only_plddt", "binder_only_pae", "binder_rmsd"):
                    values = [
                        metrics.get(f"model_{model_number}_{metric_name}")
                        for model_number in range(1, options.model_count + 1)
                    ]
                    numeric = [float(value) for value in values if value is not None]
                    metrics[f"average_{metric_name}"] = sum(numeric) / len(numeric) if numeric else None

            json.dump(metrics, handle)
            handle.write("\n")
            handle.flush()


if __name__ == "__main__":
    main()
