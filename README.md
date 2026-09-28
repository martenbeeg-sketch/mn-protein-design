# mn-protein-design

Local Streamlit workbench for Docker-orchestrated protein target preparation,
binder design, sequence optimization, refolding, analysis, and benchmarking.

## Development handoff

New development sessions should read:

1. `APP_DEVELOPMENT.md`
2. `APP_feature.md`
3. `APP_MISSING.md`
4. this README

Detailed current features are in `APP_FEATURES.md`. Engine integration guidance
is in `LLM_ENGINE_INTEGRATION_GUIDE.md`, and native container checks are in
`smoke_tests/README.md`.

## Install

Create or update the documented Conda environment:

```bash
/home/user/mambaforge/bin/conda env update -f environment.yml --prune
```

Install the package if needed:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pip install -e .
```

## Run

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design init
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design app \
  --server.address 0.0.0.0
```

The app selects the first free port beginning at 8501. To choose a different
starting port:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design app \
  --server.address 0.0.0.0 \
  --port 8508
```

## Runtime paths

Current defaults:

```text
app home:  <checkout>/mn-protein-design-workdir
runs:      <app-home>/workdir/runs
temporary: <checkout>/.tmp
references:/mnt/db/reference_files when available
```

Overrides:

```bash
export MN_PROTEIN_DESIGN_APP_HOME=/path/to/app-home
export MN_PROTEIN_DESIGN_RUN_DIR=/path/to/runs
export MN_PROTEIN_DESIGN_REFERENCE_DIR=/path/to/reference_files
export TMPDIR=/path/to/tmp
```

`Settings` shows effective runtime, reference, executable, and image paths. Some
workflow adapters still contain direct shared-reference or checkout assumptions;
see `APP_MISSING.md`.

## Main workflows

- Target Preparation
- PPI / Hotspot Detection
- Target Cropping
- Design
- Design Campaigns
- Sequence Design
- Candidate Sets
- Refolding / Validation
- Analysis
- Binder Benchmark
- Capacity Benchmark

Reusable scientific outputs are normalized under:

```text
artifacts/normalized_candidates/candidates.jsonl
```

New engine integrations must preserve explicit target/binder chain roles and
native outputs. See `LLM_ENGINE_INTEGRATION_GUIDE.md`.

## Docker and references

Scientific engines run in separate Docker images. Large model weights, MSA
databases, and caches belong under the configured reference root, normally
`/mnt/db/reference_files`.

Container build and setup notes:

```text
containers/README.md
containers/<engine>/README.md
smoke_tests/README.md
```

Build configured images with:

```bash
docker compose build
```

Run only the engine-specific native smoke needed for a change; many checks
require substantial model downloads and GPU time.

## Tests

At the 2026-07-24 audit:

- no first-party top-level `tests/` suite was present;
- no pytest configuration was present;
- `pytest` was not installed in the documented Conda environment;
- `python -m pytest` failed with `No module named pytest`.

Tests in `tools_to_implement/` and `ui_inspiration/` belong to upstream or
reference projects and are not the app regression suite.

The recommended future baseline is a CPU/local first-party pytest and Streamlit
AppTest suite, with Docker/GPU scientific smokes kept separate.

## Repository caution

Keep simulation outputs, uploaded structures, model weights, caches, and other
user-specific runtime data out of source commits. Preserve historical run
folders and local source/reference data.
