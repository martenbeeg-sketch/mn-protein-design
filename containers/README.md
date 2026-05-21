# Docker Tool Integration Notes

This directory contains Docker wrappers for tools that should become tasks in
the `mn-protein-design` Streamlit app. The overall app plan lives in
[../APP_BUILD_PLAN.md](../APP_BUILD_PLAN.md). The app should treat each
container as a tool backend with a small, explicit contract:

- one input job directory
- one output job directory
- read-only reference/model mounts from `/mnt/db/reference_files`
- GPU access only for tools that need it
- captured stdout/stderr logs per task

The real smoke-test commands live in [../smoke_tests/README.md](../smoke_tests/README.md).

## Shared Runtime Convention

Use this host-side layout for app-created jobs:

```text
/tmp/mn-protein-design-jobs/<job_id>/
  input/
  output/
  logs/
  config/
```

Recommended Docker mount convention:

```bash
-v /tmp/mn-protein-design-jobs/<job_id>:/work
-v /mnt/db/reference_files:/ref:ro
```

For tools that need writable model downloads, do a separate setup/download task
with a narrower writable mount, for example:

```bash
-v /mnt/db/reference_files/genie3:/ref/genie3
```

Then run normal jobs with references mounted read-only.

## Images

| Tool | Image | GPU | Main App Task Group | Status |
| --- | --- | --- | --- | --- |
| ScanNet | `mnprot-scannet:latest` | No | PPI / binding-site detection | Real smoke test passed |
| Genie3 | `mnprot-genie3-cu128:latest` | Yes, CUDA 12.8 | Backbone / binder design | Real generation smoke test passed |
| Surf2Spot | `mnprot-surf2spot-cu128:latest` | Yes, CUDA 12.8 | Hotspot detection | Real HS pipeline smoke test passed |
| PXDesign | `mnprot-pxdesign-cu128:latest` | Yes, CUDA 12.8 | Binder design / target preparation | Real target-parse and tiny inference smoke tests passed |
| Protpardelle-1c | `mnprot-protpardelle-1c-cu128:latest` | Yes, CUDA 12.8 | Binder backbone design / motif scaffolding | PDL1 binder backbone smoke test |

Build all images:

```bash
docker compose build
```

Build one image:

```bash
docker compose build genie3
docker compose build scannet
docker compose build surf2spot
docker compose build pxdesign
docker compose build protpardelle-1c
```

## ScanNet

Purpose in app:

- PPI/interface residue prediction
- quick target annotation before design

Image:

```text
mnprot-scannet:latest
```

Runtime:

- CPU-only
- TensorFlow 1.14 / Python 3.6 stack
- intentionally not GPU-enabled because that legacy stack is not realistic for
  RTX 5090 / Blackwell

Example app command:

```bash
docker run --rm \
  -v /tmp/mn-protein-design-jobs/<job_id>/output:/opt/ScanNet/predictions \
  mnprot-scannet:latest \
  python predict_bindingsites.py 1brs_A --noMSA
```

Expected outputs:

```text
annotated_<pdb>.pdb
predictions_<pdb>.csv
annotated_<pdb>.cxc
annotated_<pdb>.py
```

App integration idea:

- Task name: `scannet_ppi_prediction`
- Inputs: PDB ID or uploaded PDB path plus chain selection
- Outputs: residue probability CSV, annotated PDB
- Job group: `ppi_detection`

## Genie3

Purpose in app:

- unconditional backbone generation
- motif scaffolding
- binder/backbone design once target datasets are wired

Image:

```text
mnprot-genie3-cu128:latest
```

Runtime:

- GPU required for useful runs
- PyTorch 2.7.1 + CUDA 12.8
- suitable for Blackwell-class GPUs when host driver supports CUDA 12.8 runtime

Reference files:

```text
/mnt/db/reference_files/genie3/pretrained/v1/config.yaml
/mnt/db/reference_files/genie3/pretrained/v1/checkpoints/step=600000.ckpt
```

Download/setup command:

```bash
docker run --rm \
  -v /mnt/db/reference_files/genie3:/ref/genie3 \
  mnprot-genie3-cu128:latest \
  hf download yeqinglin/genie3 --include 'pretrained/**' --local-dir /ref/genie3
```

Example app command:

```bash
docker run --rm --gpus all \
  -v /mnt/db/reference_files/genie3/pretrained:/opt/genie3/pretrained:ro \
  -v /tmp/mn-protein-design-jobs/<job_id>:/work \
  mnprot-genie3-cu128:latest \
  genie3 generate -c /work/config/experiment.yaml --num-devices 1
```

Expected outputs:

```text
output/<experiment>/pdbs/*.pdb
output/<experiment>/generation_stats/*.json
```

App integration idea:

- Task name: `genie3_generate`
- Inputs: experiment YAML or app-generated config fields
- Outputs: generated PDBs, generation stats
- Job group: `design`

## Protpardelle-1c

Purpose in app:

- PDL1-style motif-conditioned binder backbone generation
- unconditional backbone generation
- motif scaffolding jobs that can later feed sequence design/refolding tasks

Image:

```text
mnprot-protpardelle-1c-cu128:latest
```

Runtime:

- GPU required for useful sampling
- PyTorch 2.7.1 + CUDA 12.8
- suitable for RTX 5090 / Blackwell when the host NVIDIA driver supports the
  CUDA 12.8 runtime
- the container sets `TORCH_CUDA_ARCH_LIST` through `12.0` for any local CUDA
  extension builds

Reference files:

```text
/mnt/db/reference_files/protpardelle-1c/model_params/configs/
/mnt/db/reference_files/protpardelle-1c/model_params/weights/
/mnt/db/reference_files/protpardelle-1c/model_params/ESMFold/
/mnt/db/reference_files/protpardelle-1c/model_params/ProteinMPNN/vanilla_model_weights/
/mnt/db/reference_files/protpardelle-1c/model_params/LigandMPNN/
```

The backbone-only smoke uses `--num-mpnn-seqs 0`, so it only needs the
Protpardelle model configs and weights. The full upstream download script also
adds ESMFold, ProteinMPNN, and LigandMPNN weights for later sequence/refold
analysis tasks.

Download/setup command:

```bash
docker run --rm \
  -v /mnt/db/reference_files/protpardelle-1c:/ref/protpardelle-1c \
  mnprot-protpardelle-1c-cu128:latest \
  bash -lc 'set -euo pipefail; cd /ref/protpardelle-1c; /opt/protpardelle-1c/download_model_params.sh'
```

Example app command:

```bash
docker run --rm --gpus all \
  -v /mnt/db/reference_files/protpardelle-1c:/ref/protpardelle-1c:ro \
  -v /tmp/mn-protein-design-jobs/<job_id>:/work \
  mnprot-protpardelle-1c-cu128:latest \
  python -m protpardelle.sample /work/config/protpardelle_pdl1.yaml \
    --motif-dir /opt/protpardelle-1c/examples/motifs/bindcraft \
    --num-samples 1 \
    --num-mpnn-seqs 0 \
    --batch-size 1 \
    --seed 7
```

Expected outputs:

```text
output/protpardelle_pdl1/*/17_PDL1(AAV)/sample_17_PDL1(AAV)_0.pdb
output/protpardelle_pdl1/*/17_PDL1(AAV)/scaffold_info.csv
output/protpardelle_pdl1/*/design_input.csv
```

App integration idea:

- Task name: `protpardelle_sample`
- Inputs: sampling YAML, motif source, number of backbone samples, batch size,
  optional seed, optional MPNN/refold count
- Outputs: sampled PDB backbones, scaffold placement CSV, design input CSV,
  task log
- Job group: `design`

## Surf2Spot

Purpose in app:

- PPI hotspot detection
- target surface and hotspot clustering for design anchors

Image:

```text
mnprot-surf2spot-cu128:latest
```

Runtime:

- GPU used for Chainsaw, ProtT5, and model inference
- PyTorch 2.7.1 + CUDA 12.8
- includes MSMS, APBS, pdb2pqr, multivalue, RDKit, PyMOL Python module, a
  narrow PyMESH compatibility layer, and bundled HS/NB Surf2Spot model weights

Reference files:

```text
/mnt/db/reference_files/surf2spot/chainsaw/saved_models/
/mnt/db/reference_files/surf2spot/model_emb/prot_t5_xl_half_uniref50-enc/
```

Download/setup commands:

```bash
docker run --rm \
  -v /mnt/db/reference_files/surf2spot:/ref/surf2spot \
  mnprot-surf2spot-cu128:latest \
  bash -lc 'set -e; tmp=$(mktemp -d); git clone --depth 1 https://github.com/JudeWells/Chainsaw "$tmp/Chainsaw"; cp -a "$tmp/Chainsaw/saved_models/." /ref/surf2spot/chainsaw/saved_models/'

docker run --rm \
  -v /mnt/db/reference_files/surf2spot/model_emb/prot_t5_xl_half_uniref50-enc:/ref/prot_t5 \
  mnprot-surf2spot-cu128:latest \
  hf download Rostlab/prot_t5_xl_half_uniref50-enc --local-dir /ref/prot_t5
```

Example app commands:

```bash
docker run --rm --gpus all \
  -v /tmp/mn-protein-design-jobs/<job_id>:/work \
  -v /mnt/db/reference_files/surf2spot/chainsaw/saved_models:/opt/Surf2Spot/Surf2Spot/data/chainsaw/saved_models:ro \
  mnprot-surf2spot-cu128:latest \
  Surf2Spot HS-preprocess -i /work/input -o /work/output/preprocess

docker run --rm --gpus all \
  -v /tmp/mn-protein-design-jobs/<job_id>:/work \
  -v /mnt/db/reference_files/surf2spot/chainsaw/saved_models:/opt/Surf2Spot/Surf2Spot/data/chainsaw/saved_models:ro \
  -v /mnt/db/reference_files/surf2spot/model_emb/prot_t5_xl_half_uniref50-enc:/opt/Surf2Spot/Surf2Spot/data/model_emb/prot_t5_xl_half_uniref50-enc:ro \
  mnprot-surf2spot-cu128:latest \
  Surf2Spot HS-craft -i /work/output/preprocess

docker run --rm --gpus all \
  -v /tmp/mn-protein-design-jobs/<job_id>:/work \
  -v /mnt/db/reference_files/surf2spot/chainsaw/saved_models:/opt/Surf2Spot/Surf2Spot/data/chainsaw/saved_models:ro \
  -v /mnt/db/reference_files/surf2spot/model_emb/prot_t5_xl_half_uniref50-enc:/opt/Surf2Spot/Surf2Spot/data/model_emb/prot_t5_xl_half_uniref50-enc:ro \
  mnprot-surf2spot-cu128:latest \
  Surf2Spot HS-predict -i /work/output/preprocess -o /work/output/predict --model /opt/Surf2Spot/model/HS/model.pt
```

Implementation notes:

The upstream Surf2Spot code expected a separate `surf2spot_tools` Conda
environment for MaSIF surface generation. The container now runs surface
generation in the main CUDA 12.8 environment. Because old `pymesh2` does not
compile cleanly on the Python 3.11 stack, the image includes a small
compatibility module for the subset of PyMESH APIs used by Surf2Spot's HS
pipeline. Keep this in mind if you later compare exact surfaces against the
original legacy PyMESH environment.

App integration idea:

- Task name: `surf2spot_hotspots`
- Inputs: uploaded PDB directory, split mode, probe radius, threshold
- Outputs: hotspot CSV, predicted PLY, and PyMOL `.pse` files
- Job group: `hotspot_detection`

## PXDesign

Purpose in app:

- target crop/hotspot input preparation
- diffusion-based binder design
- later full pipeline ranking with AF2 / Protenix evaluation when the larger
  evaluation references are mounted

Image:

```text
mnprot-pxdesign-cu128:latest
```

Runtime:

- GPU required for useful design runs
- PyTorch 2.7.1 + CUDA 12.8 for Blackwell-class GPUs
- installs PXDesign, Protenix v0.5.0+pxd, PXDesignBench, ColabDesign, JAX CUDA,
  DeepSpeed, and CUTLASS
- Protenix is installed with `--no-deps` so it does not downgrade the CUDA 12.8
  PyTorch stack to its upstream CUDA 12.1 pin

Reference files:

```text
/mnt/db/reference_files/pxdesign/release_data/ccd_cache/components.v20240608.cif
/mnt/db/reference_files/pxdesign/release_data/ccd_cache/components.v20240608.cif.rdkit_mol.pkl
/mnt/db/reference_files/pxdesign/release_data/ccd_cache/clusters-by-entity-40.txt
/mnt/db/reference_files/pxdesign/release_data/checkpoint/*.pt
/mnt/db/reference_files/pxdesign/tool_weights/af2/*.npz
/mnt/db/reference_files/pxdesign/tool_weights/mpnn/
```

The target-parse smoke only needs the CCD cache. The tiny `infer` smoke
downloaded PXDesign and Protenix checkpoints into `release_data/checkpoint` by
passing that directory as `--load_checkpoint_dir`. Full `pipeline` also needs
AF2 and MPNN tool weights under `tool_weights`.

CCD cache setup command:

```bash
mkdir -p /mnt/db/reference_files/pxdesign/release_data/ccd_cache
docker run --rm \
  -v /mnt/db/reference_files/pxdesign/release_data/ccd_cache:/ref/pxdesign/release_data/ccd_cache \
  mnprot-pxdesign-cu128:latest \
  bash -lc 'set -euo pipefail; cd /ref/pxdesign/release_data/ccd_cache; for url in https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl https://pxdesign.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt; do file=$(basename "$url"); if [ ! -s "$file" ]; then curl -L -C - "$url" -o "$file"; fi; done'
```

Example target-parse command:

```bash
docker run --rm \
  -v /tmp/mn-protein-design-jobs/<job_id>:/work \
  -v /mnt/db/reference_files/pxdesign:/ref/pxdesign:ro \
  mnprot-pxdesign-cu128:latest \
  pxdesign parse-target --yaml /work/config/task.yaml -o /work/output/parse-target
```

Example design command:

```bash
docker run --rm --gpus all \
  -v /tmp/mn-protein-design-jobs/<job_id>:/work \
  -v /mnt/db/reference_files/pxdesign:/ref/pxdesign \
  mnprot-pxdesign-cu128:latest \
  pxdesign infer \
    -i /work/config/task.yaml \
    -o /work/output/infer \
    --N_sample 1 \
    --N_step 50 \
    --dtype bf16 \
    --load_checkpoint_dir /ref/pxdesign/release_data/checkpoint
```

Expected target-parse outputs:

```text
*_parsed_target.cif
*_parsed_target.pml
tmp/*.json
tmp/*.pkl.gz
```

App integration idea:

- Task name: `pxdesign_parse_target`
- Inputs: uploaded PDB/CIF, chain crop, hotspot residues, binder length
- Outputs: parsed target CIF, PyMOL helper script, normalized JSON
- Job group: `target_input_structure_generation`
- Task name: `pxdesign_infer`
- Inputs: validated PXDesign YAML, sample count, diffusion steps, dtype
- Outputs: generated binder structures and inference metadata
- Job group: `design`

## App-Orchestration Shape

The Streamlit app should not hardcode shell snippets directly in UI pages.
Instead, keep a registry like:

```python
TOOL_REGISTRY = {
    "scannet_ppi_prediction": {
        "image": "mnprot-scannet:latest",
        "gpu": False,
        "group": "ppi_detection",
        "reference_mounts": [],
    },
    "genie3_generate": {
        "image": "mnprot-genie3-cu128:latest",
        "gpu": True,
        "group": "design",
        "reference_mounts": [
            "/mnt/db/reference_files/genie3/pretrained:/opt/genie3/pretrained:ro",
        ],
    },
    "surf2spot_hotspots": {
        "image": "mnprot-surf2spot-cu128:latest",
        "gpu": True,
        "group": "hotspot_detection",
        "reference_mounts": [
            "/mnt/db/reference_files/surf2spot/chainsaw/saved_models:/opt/Surf2Spot/Surf2Spot/data/chainsaw/saved_models:ro",
            "/mnt/db/reference_files/surf2spot/model_emb/prot_t5_xl_half_uniref50-enc:/opt/Surf2Spot/Surf2Spot/data/model_emb/prot_t5_xl_half_uniref50-enc:ro",
        ],
    },
    "pxdesign_parse_target": {
        "image": "mnprot-pxdesign-cu128:latest",
        "gpu": False,
        "group": "target_input_structure_generation",
        "reference_mounts": [
            "/mnt/db/reference_files/pxdesign:/ref/pxdesign:ro",
        ],
    },
    "pxdesign_infer": {
        "image": "mnprot-pxdesign-cu128:latest",
        "gpu": True,
        "group": "design",
        "reference_mounts": [
            "/mnt/db/reference_files/pxdesign:/ref/pxdesign",
        ],
    },
    "protpardelle_sample": {
        "image": "mnprot-protpardelle-1c-cu128:latest",
        "gpu": True,
        "group": "design",
        "reference_mounts": [
            "/mnt/db/reference_files/protpardelle-1c:/ref/protpardelle-1c:ro",
        ],
    },
}
```

The job runner can then add:

- `--gpus all` when `gpu=True`
- `/work` mount for the job directory
- reference mounts from the registry
- a log capture wrapper around Docker stdout/stderr
