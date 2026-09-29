#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch
from esm.models.esmc import EsmcForMaskedLM


def _load_tutorial(path: Path):
    spec = importlib.util.spec_from_file_location("biohub_binder_design", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load binder design tutorial: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_model(module: Any, model_dir: Path, esmc_dir: Path) -> Any:
    if not model_dir.exists():
        raise FileNotFoundError(f"Experimental ESMFold2 checkpoint is missing: {model_dir}")
    model = module.EsmFold2ExperimentalModel.from_pretrained(
        str(model_dir),
        load_esmc=False,
        local_files_only=True,
        device="cuda",
    )
    model.load_esmc(str(esmc_dir))
    model.configure_lm_dropout(0.5, force_lm_dropout_during_inference=True)
    backend = None
    if module.TRITON_KERNELS_AVAILABLE:
        backend = module.BACKEND_FUSED
    elif module.CUE_AVAILABLE:
        backend = module.BACKEND_CUEQ
    model.set_kernel_backend(backend)
    return model.cuda().eval().requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Biohub ESMFold2 binder design")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_root = Path(config["reference_root"])

    tutorial = _load_tutorial(Path(config["tutorial_path"]))
    tutorial.STEPS = int(config["optimization_steps"])
    tutorial.LEARNING_RATE = float(config["learning_rate"])
    tutorial.LOG_INTERVAL = int(config.get("log_interval", 5))
    tutorial.REUSE_ESMC = True
    tutorial.COMPILE = bool(config.get("compile", False))
    tutorial.CHECKPOINT_LM = bool(config.get("checkpoint_lm", False))

    model_name = str(config["model_name"])
    shared_model = _load_model(
        tutorial,
        reference_root / "binder-design" / model_name,
        reference_root / "ESMC-6B",
    )
    esmc_model = EsmcForMaskedLM.from_pretrained(
        str(reference_root / "ESMC-6B"),
        dtype=torch.float32,
        device="cpu",
    )
    del esmc_model.esmc
    torch.cuda.empty_cache()
    esmc_model.esmc = shared_model.esmc
    esmc_model = esmc_model.cuda().eval().requires_grad_(False)

    engine = tutorial.ESMFold2Design()
    engine.inversion_models = {model_name: shared_model}
    engine.hf_critic_models = {model_name: shared_model}
    engine.esmc_model = esmc_model

    target_sequence = str(config["target_sequence"]).strip().upper()
    binder_prompt = "#" * int(config["binder_length"])
    designs: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    fasta_lines: list[str] = []

    for design_index in range(1, int(config["num_designs"]) + 1):
        seed = int(config["seed"]) + design_index - 1
        sequences, trajectory, critic_results = engine.design(
            target_sequence=target_sequence,
            binder_sequence=binder_prompt,
            is_antibody=False,
            seed=seed,
            batch_size=1,
        )
        binder_sequence = sequences[0].split("|")[-1]
        fasta_lines.extend([f">esmfold2_design_{design_index:05d}", binder_sequence])

        for step, losses in sorted(trajectory.items()):
            trajectory_rows.append(
                {
                    "design_index": design_index,
                    "step": int(step),
                    **{name: _mean(value) for name, value in losses.items()},
                }
            )

        critics: list[dict[str, Any]] = []
        for critic_index, result in enumerate(critic_results, start=1):
            complex_path = output_dir / (
                f"esmfold2_design_{design_index:05d}_critic_{critic_index:02d}.cif"
            )
            if result.get("complex") is not None:
                complex_path.write_text(result["complex"].to_mmcif_string())
            critics.append(
                {
                    "critic_name": result.get("critic_name"),
                    "final_loss": _finite_or_none(result.get("final_loss")),
                    "iptm": _finite_or_none(result.get("iptm")),
                    "distogram_iptm_proxy": _finite_or_none(
                        result.get("distogram_iptm_proxy")
                    ),
                    "cdr_distogram_iptm_proxy": _finite_or_none(
                        result.get("cdr_distogram_iptm_proxy")
                    ),
                    "complex_path": str(complex_path) if complex_path.exists() else None,
                }
            )
        designs.append(
            {
                "design_index": design_index,
                "seed": seed,
                "binder_sequence": binder_sequence,
                "binder_length": len(binder_sequence),
                "critics": critics,
            }
        )

    (output_dir / "designed_binders.fasta").write_text("\n".join(fasta_lines) + "\n")
    (output_dir / "design_metrics.json").write_text(
        json.dumps(
            {
                "profile": "lean_shared_model",
                "model": model_name,
                "optimization_steps": tutorial.STEPS,
                "designs": designs,
            },
            indent=2,
        )
        + "\n"
    )
    if trajectory_rows:
        with (output_dir / "optimization_trajectory.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
            writer.writeheader()
            writer.writerows(trajectory_rows)


if __name__ == "__main__":
    main()
