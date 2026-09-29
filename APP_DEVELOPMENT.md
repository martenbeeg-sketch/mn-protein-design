# mn-protein-design Development Handoff

Updated: 2026-09-29

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

`mn-protein-design install-launchers` writes per-user terminal wrappers and an
optional desktop shortcut through `mn_protein_design/launchers.py`. The app
wrapper records the active Python executable and effective runtime paths so
it can start without shell activation. These generated files belong to the
machine user's home directory, not source control; reinstall them after moving
the environment or changing runtime paths.

- app home: `MN_PROTEIN_DESIGN_APP_HOME`, currently defaulting to
  `<checkout>/mn-protein-design-workdir`;
- runs: `MN_PROTEIN_DESIGN_RUN_DIR`, otherwise `<app-home>/workdir/runs`;
- references: `MN_PROTEIN_DESIGN_REFERENCE_DIR`, otherwise
  `/mnt/db/reference_files` when present;
- temporary files: `TMPDIR`, otherwise `<checkout>/.tmp`.

The current defaults remain checkout-coupled. Do not move historical run folders
or rewrite their metadata merely to improve portability. `core/portable_paths.py`
maps managed run, reference, and app-home files to stable URI prefixes when
new job JSON and candidate structures are serialized, and resolves those URIs
or run-relative paths against the active machine's runtime roots. It also
remaps selected legacy absolute paths after their original location disappears.
`mn-protein-design portability` audits, copies, verifies, and imports the full
workdir plus configured runs without changing the source. Reference files and
Docker images remain separate. Historical records are normalized only in the
exported copy; arbitrary required paths outside managed roots still stop export.

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
- `mn_protein_design/core/local_worker.py`: worker-request dispatch and queue
  service startup.
- `mn_protein_design/core/scheduler.py`: persistent local queue service, CPU
  slot allocation, per-GPU admission, health snapshots, and Docker CPU limits.
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

The project provides `pytest` as an optional development dependency and keeps
first-party tests under `tests/`. Install it with:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pip install -e '.[dev]'
```

Run the CPU/local suite with:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pytest -q
```

The suite currently collects 361 tests across file-backed jobs and scheduler
behavior, workflow automation, portability and migration, target-fragment
preparation, candidate imports, chain-role validation, design parameters,
benchmark metrics, and runtime estimates. Streamlit AppTests cover startup for
the main pages and selection of lazy result/setup panels. Tests use temporary
run roots and small synthetic structures; they do not launch design or
refolding engines. Native Docker/GPU checks remain in `smoke_tests/`.

Tests under `tools_to_implement/` and `ui_inspiration/` belong to upstream or
reference projects and are not the app regression suite. Container/GPU smoke
tests remain separate from the local test baseline.

## Safe development procedure

1. Inspect the dirty worktree and preserve unrelated changes.
2. Read the relevant workflow, page, normalized-candidate code, and chain-role
   helpers before editing.
3. Keep engine-native layouts inside adapters and normalize at boundaries.
4. Preserve historical runs and legacy candidate compatibility.
5. Add focused unit tests and Streamlit AppTests before expanding UI claims.
6. Run the first-party pytest suite.
7. For scientific adapters, separately run the documented native Docker/GPU
   smoke and verify non-empty normalized candidates and downstream handoff.
8. Update these four handoff files when architecture or validation status
   changes.

Queued jobs with `worker_request.json` are dispatched by a separate worker
service. Heavy design, refolding, sequence-design, analysis, detection, and
benchmark launches use this queue. The service starts automatically when a UI
path enqueues a worker job, or can run in a terminal with
`mn-protein-design worker`. It keeps running when Streamlit exits. The default
worker capacity is up to 32 CPU slots and can be changed with
`MN_PROTEIN_DESIGN_WORKER_CPU_SLOTS`; CPU-only and GPU jobs reserve CPU slots,
and GPU jobs also reserve their selected device. Docker receives the slot count
as thread settings and uses a CPU quota when the host exposes cgroup quota
controls. Child steps in a workflow share the parent allocation.

## Near-term engineering direction

The app does not need a wholesale rewrite. Its scientific breadth and normalized
candidate model are valuable. The most useful future work is consolidation:

1. extend first-party tests across workflows, resource admission, and pages;
2. strengthen run-relative typed artifacts and path portability;
3. migrate remaining synchronous workflows into the background queue;
4. add RAM/VRAM/scratch admission and cleaner worker crash recovery;
5. split oversized workflow/result modules along stable contracts;
6. make engine manifests, references, resources, licenses, and validation status
   authoritative rather than distributed across constants and documentation.
