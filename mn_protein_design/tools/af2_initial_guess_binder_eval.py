#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
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
    parser.add_argument("--use-binder-template", action="store_true", default=False)
    parser.add_argument("--use-interface-template", action="store_true", default=False)
    parser.add_argument("--blind", action="store_true", default=False)
    parser.add_argument("--multimer", action="store_true", default=False)
    parser.add_argument("--designed_chains", default="A")
    parser.add_argument("--cyclic", action="store_true", default=False)
    parser.add_argument("--hotspot", default=None)
    options = parser.parse_args()

    if options.blind:
        assert not options.use_binder_template
        assert not options.use_interface_template
    if options.use_interface_template:
        assert options.use_binder_template

    model = mk_af_model(
        protocol="binder",
        data_dir=options.params,
        use_multimer=options.multimer,
        model_names=["model_1_multimer_v3" if options.multimer else "model_1_ptm"],
        use_initial_guess=not options.blind,
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
            target_chain = "B" if options.designed_chains == "A" else "A"
            if options.designed_chains not in {"A", "B"}:
                raise NotImplementedError("Expected binder chain to be A or B")

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
            model.set_seq(mode="wildtype")
            model.set_opt(num_recycles=options.num_recycles)
            model.predict(num_models=1, verbose=False)

            metrics = {"id": basename}
            metrics.update({new_key: model.aux["log"].get(old_key) for old_key, new_key in METRICS.items()})
            metrics["binder_plddt"] *= 100
            metrics["binder_pae"] *= 31.0
            metrics["ipae"] *= 31.0
            metrics["time"] = time.time() - start_time

            ca_pos = model.aux["atom_positions"][:, 1]
            ca_dist = np.sqrt(np.square(ca_pos[model._target_len :, None] - ca_pos[None, : model._target_len]).sum(axis=-1) + 1e-8)
            target_interface_res = model.aux["residue_index"][: model._target_len][ca_dist.min(axis=0) <= 8]
            metrics["interface_target_residues"] = ",".join(f"B{pos}" for pos in target_interface_res)

            json.dump(metrics, handle)
            handle.write("\n")
            handle.flush()

            suffix = os.path.basename(options.output_name.rstrip("/"))
            out_base = os.path.join(options.output_name, f"{basename}_{suffix}")
            save_binder_design_pdb(model, f"{out_base}.pdb")
            _save_pae_json(model, f"{out_base}_pae.json", metrics)


if __name__ == "__main__":
    main()
