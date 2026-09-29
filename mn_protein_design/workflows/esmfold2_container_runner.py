#!/usr/bin/env python
"""Run a batch of ESMFold2 predictions from a job-local JSON request."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.")
    if not text:
        raise ValueError("Every ESMFold2 request needs a non-empty request_id.")
    return text[:180]


def _numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype or np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    elif hasattr(value, "item"):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_model(reference_root: Path, device: str) -> Any:
    import torch

    model_dir = reference_root / "ESMFold2"
    esmc_dir = reference_root / "ESMC-6B"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"ESMFold2 weights are missing: {model_dir}")
    if not esmc_dir.is_dir():
        raise FileNotFoundError(f"ESMC-6B weights are missing: {esmc_dir}")
    if device != "cuda":
        raise RuntimeError("The Biohub ESMFold2 runtime requires CUDA; select a GPU for this job.")
    if not torch.cuda.is_available():
        raise RuntimeError("This ESMFold2 job requested CUDA, but CUDA is unavailable in the container.")
    from esm.models.esmfold2 import EsmFold2Model

    model = EsmFold2Model.from_pretrained(
        str(model_dir), load_esmc=False, device=device
    )
    model.load_esmc(str(esmc_dir), precision="bf16")
    return model.to(device).eval()


def _prediction_input(request: dict[str, Any]) -> StructurePredictionInput:
    from esm.models.esmfold2 import (
        DistogramConditioning,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.utils.msa import MSA

    sequences: list[ProteinInput] = []
    for chain in request.get("sequences") or []:
        msa_path = str(chain.get("msa_path") or "").strip()
        msa = MSA.from_a3m(Path(msa_path)) if msa_path else None
        sequences.append(
            ProteinInput(
                id=str(chain["id"]),
                sequence=str(chain["sequence"]),
                msa=msa,
            )
        )
    if not sequences:
        raise ValueError("An ESMFold2 prediction request must contain at least one protein sequence.")
    conditioning: list[DistogramConditioning] = []
    for item in request.get("distogram_conditioning") or []:
        values = np.load(Path(item["path"]), allow_pickle=False)
        conditioning.append(DistogramConditioning(chain_id=str(item["chain_id"]), distogram=values))
    return StructurePredictionInput(
        sequences=sequences,
        distogram_conditioning=conditioning or None,
    )


def _write_result(result: Any, request_id: str, output_dir: Path) -> dict[str, Any]:
    request_dir = output_dir / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    complex_path = request_dir / "complex.cif"
    complex_path.write_text(result.complex.to_mmcif())
    arrays_path = request_dir / "result_arrays.npz"
    np.savez_compressed(
        arrays_path,
        atom_positions=_numpy(result.complex.atom_positions, dtype=np.float32),
        atom_elements=_numpy(result.complex.atom_elements, dtype=str),
        atom_names=_numpy(result.complex.atom_names, dtype=str),
        token_to_atoms=_numpy(result.complex.token_to_atoms, dtype=np.int32),
        chain_id=_numpy(result.complex.chain_id, dtype=np.int32),
        plddt=_numpy(result.plddt, dtype=np.float32),
        pae=_numpy(result.pae, dtype=np.float32),
        distogram=_numpy(result.distogram, dtype=np.float32),
        pair_chains_iptm=_numpy(result.pair_chains_iptm, dtype=np.float32),
    )
    metadata = result.complex.metadata
    return {
        "request_id": request_id,
        "complex_path": str(complex_path.relative_to(Path("/work"))),
        "arrays_path": str(arrays_path.relative_to(Path("/work"))),
        "complex": {
            "id": str(result.complex.id),
            "sequence": [str(sequence) for sequence in result.complex.sequence],
            "entity_lookup": {str(key): str(value) for key, value in metadata.entity_lookup.items()},
            "chain_lookup": {str(key): str(value) for key, value in metadata.chain_lookup.items()},
        },
        "ptm": _finite(result.ptm),
        "iptm": _finite(result.iptm),
    }


def run(config_path: Path) -> None:
    import torch

    config = json.loads(config_path.read_text())
    work_root = Path(config.get("work_root") or "/work")
    reference_root = Path(config.get("reference_root") or "/ref/biohub-esm")
    output_dir = Path(config.get("output_dir") or "/work/artifacts/raw/esmfold2_runtime/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = str(config.get("device") or "cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("The Biohub ESMFold2 runtime requires CUDA; select a GPU for this job.")
    model = _load_model(reference_root, device)
    from esm.models.esmfold2 import ESMFold2InputBuilder

    builder = ESMFold2InputBuilder(ccd_cache=reference_root / "ESMFold2")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    requests = config.get("requests") or []
    for index, request in enumerate(requests, start=1):
        request_id = _safe_name(request.get("request_id") or f"prediction_{index:05d}")
        if request_id in seen:
            raise ValueError(f"Duplicate ESMFold2 request_id: {request_id}")
        seen.add(request_id)
        print(f"Folding {request_id} ({index}/{len(requests)})", flush=True)
        result = builder.fold(
            model,
            _prediction_input(request),
            num_loops=int(request.get("num_loops", config.get("num_loops", 3))),
            num_sampling_steps=int(request.get("num_sampling_steps", config.get("num_sampling_steps", 32))),
            num_diffusion_samples=int(request.get("num_diffusion_samples", 1)),
            seed=int(request.get("seed", config.get("seed", 0))),
            complex_id=request_id,
        )
        results.append(_write_result(result, request_id, output_dir))
        del result
        if device == "cuda":
            torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "device": device,
        "num_loops": int(config.get("num_loops", 3)),
        "num_sampling_steps": int(config.get("num_sampling_steps", 32)),
        "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(results)} ESMFold2 predictions on {device}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
