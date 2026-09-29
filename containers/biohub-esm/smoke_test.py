#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the Biohub ESM CUDA runtime.")
    parser.add_argument(
        "--reference-root",
        default="/ref/biohub-esm",
        help="Mounted Biohub ESM reference directory.",
    )
    parser.add_argument(
        "--load-esmfold2",
        action="store_true",
        help="Also load ESMFold2 weights. This is heavier than the default import check.",
    )
    args = parser.parse_args()

    reference_root = Path(args.reference_root)
    esmfold2_dir = reference_root / "ESMFold2"
    esmc_dir = reference_root / "ESMC-6B"
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        print(f"cuda_capability={torch.cuda.get_device_capability(0)}")
    print(f"reference_root={reference_root}")
    print(f"esmfold2_dir_exists={esmfold2_dir.exists()}")
    print(f"esmc_dir_exists={esmc_dir.exists()}")

    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        EsmFold2Model,
        ProteinInput,
        StructurePredictionInput,
    )

    _ = ESMFold2InputBuilder(ccd_cache=esmfold2_dir)
    _ = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="ACDEFGHIK")])
    print("esmfold2_imports=ok")

    if args.load_esmfold2:
        if not esmfold2_dir.exists():
            raise FileNotFoundError(f"ESMFold2 directory is missing: {esmfold2_dir}")
        if not esmc_dir.exists():
            raise FileNotFoundError(f"ESMC-6B directory is missing: {esmc_dir}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EsmFold2Model.from_pretrained(
            str(esmfold2_dir), load_esmc=False, device=device
        )
        model.load_esmc(str(esmc_dir), precision="bf16")
        model = model.eval()
        print(f"esmfold2_loaded=ok device={model.device}")


if __name__ == "__main__":
    main()
