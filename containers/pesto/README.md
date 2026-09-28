# PeSTo container

This image exposes PeSTo i_v4_1 protein-interface inference as a deterministic
CLI for the app's queued PPI detection workflow.

The image reuses the app's CUDA 12.8 / PyTorch 2.7 Biohub base and does not bake
model parameters into an image layer. The checkpoint must exist at:

```text
/mnt/db/reference_files/pesto/i_v4_1/model_ckpt.pt
```

Build it with:

```bash
docker compose build pesto
```

The normalized output table contains `chain`, `residue`, `amino_acid`,
`residue_name`, and `score`. Chain IDs and residue numbers match the staged
input PDB.
