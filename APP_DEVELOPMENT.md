# mn-protein-design Development Handoff

Updated: 2026-09-28

This is the starting point for future development sessions. Read:

1. `APP_DEVELOPMENT.md` for engineering conventions and verification.
2. `APP_feature.md` for the current product and conceptual architecture.
3. `APP_MISSING.md` for gaps and possible future improvements.
4. `README.md` for installation and startup.

`APP_FEATURES.md`, `LLM_ENGINE_INTEGRATION_GUIDE.md`, the container READMEs, and
`smoke_tests/README.md` remain useful detailed references. Current code takes
precedence when older documentation disagrees with it.

## Scope and neighboring application

`mn-protein-design` is a protein binder-design, refolding, evaluation, and
benchmarking application. Its central reusable entity is a normalized protein
candidate with explicit target and binder chain roles.

`mn-ligand` is a separate ligand-centric application. Generic engineering ideas
may be shared, but neither app should import the other at runtime or share job
state. Protein candidate semantics, chain-role conventions, binder benchmarks,
and design campaigns do not belong in mn-ligand unchanged.

## Current repository state

The current application, workflow, container, and documentation source snapshot
is committed. Preserve historical run folders and local source/reference data,
but keep simulation outputs, uploaded structures, model weights, and caches out
of source commits.

The current implementation is much larger than the committed baseline. Important
new or heavily changed areas include design campaigns, capacity benchmarking,
candidate imports, chain-role normalization, target preparation and analysis,
refolding engines, benchmark result handling, runtime estimation, and queued
local-worker execution.

## Runtime contract

Runtime resolution is owned by `mn_protein_design/runtime.py`.

- app home: `MN_PROTEIN_DESIGN_APP_HOME`, currently defaulting to
  `<checkout>/mn-protein-design-workdir`;
- runs: `MN_PROTEIN_DESIGN_RUN_DIR`, otherwise `<app-home>/workdir/runs`;
- references: `MN_PROTEIN_DESIGN_REFERENCE_DIR`, otherwise
  `/mnt/db/reference_files` when present;
- temporary files: `TMPDIR`, otherwise `<checkout>/.tmp`.

The current defaults remain checkout-coupled. Do not move historical run folders
or rewrite their metadata merely to improve portability. Add compatibility
resolution before changing stored path behavior.

## Job and candidate contracts

Runs use a file-backed layout:

```text
runs/<task-group>/<run-id>/
  input.json
  metadata.json
  command.json
  stdout.log
  stderr.log
  result.json
  worker_request.json        # queued local-worker jobs where applicable
  artifacts/
```

The main cross-workflow scientific exchange format is:

```text
artifacts/normalized_candidates/candidates.jsonl
artifacts/normalized_candidates/campaign_result.json
```

Candidate definitions and normalization live in
`mn_protein_design/core/candidates.py`. New design/refolding integrations should
normalize their outputs at workflow boundaries and preserve native outputs
separately.

Explicit chain roles are a core scientific contract:

- target/reference chains are staged from `A`, `B`, `C`, ...;
- binder/design chains are staged from `Z`, `Y`, `X`, ...;
- engine-local compatibility layouts are allowed only inside the adapter;
- normalized outputs must restore explicit biological roles.

Use `mn_protein_design/workflows/chain_roles.py`. Do not infer roles from chain
alphabetical order or restore the historical binder-`A` assumption globally.

## Main implementation landmarks

- `mn_protein_design/core/jobs.py`: file-backed jobs, statuses, queue/resource
  locks, pause/stop/resume, and deletion helpers.
- `mn_protein_design/core/candidates.py`: normalized candidate contract,
  structure handling, and candidate metrics.
- `mn_protein_design/core/local_worker.py`: worker-request dispatch and spawned
  background execution.
- `mn_protein_design/core/runtime_estimator.py`: history-based runtime estimates.
- `mn_protein_design/workflows/chain_roles.py`: explicit chain-role schema.
- `mn_protein_design/workflows/design.py`: individual design-engine adapters.
- `mn_protein_design/workflows/design_campaigns.py`: multi-engine staged design
  campaigns.
- `mn_protein_design/workflows/refolding.py`: folding/refolding adapters.
- `mn_protein_design/workflows/benchmark.py`: binder benchmark orchestration and
  metric aggregation.
- `mn_protein_design/workflows/capacity_benchmark.py`: capacity-test scheduling
  and aggregation.
- `mn_protein_design/app/pages/results.py`: shared but very large result surface.
- `mn_protein_design/app/pages/settings.py`: current read-only installation/path
  inventory and migration notes.

Use `LLM_ENGINE_INTEGRATION_GUIDE.md` when adding or repairing a prediction
engine.

## Python environment and tests

Use:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python
```

The environment currently does **not** contain `pytest`, and no first-party
top-level `tests/` suite or pytest configuration was found during the
2026-07-24 audit. Running:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pytest
```

currently fails with `No module named pytest`.

Tests found under `tools_to_implement/` and `ui_inspiration/` belong to vendored
or reference projects and must not be treated as the mn-protein-design
regression suite. Container and workflow smoke procedures are documented in
`smoke_tests/README.md`, but they are not a replacement for local unit and
Streamlit AppTest coverage.

The first testing improvement should create a first-party `tests/` directory,
add a development dependency, and establish a CPU/local baseline that does not
require Docker, a GPU, model weights, or network access.

## Safe development procedure

1. Inspect the dirty worktree and preserve unrelated changes.
2. Read the relevant workflow, page, normalized-candidate code, and chain-role
   helpers before editing.
3. Keep engine-native layouts inside adapters and normalize at boundaries.
4. Preserve historical runs and legacy candidate compatibility.
5. Add focused unit tests and Streamlit AppTests before expanding UI claims.
6. Run the first-party suite once it exists.
7. For scientific adapters, separately run the documented native Docker/GPU
   smoke and verify non-empty normalized candidates and downstream handoff.
8. Update these four handoff files when architecture or validation status
   changes.

## Near-term engineering direction

The app does not need a wholesale rewrite. Its scientific breadth and normalized
candidate model are valuable. The most useful future work is consolidation:

1. establish the missing first-party test suite;
2. strengthen run-relative typed artifacts and path portability;
3. split oversized workflow/result modules along stable contracts;
4. converge synchronous and spawned-worker execution on one durable worker
   lifecycle;
5. make engine manifests, references, resources, licenses, and validation status
   authoritative rather than distributed across constants and documentation.
