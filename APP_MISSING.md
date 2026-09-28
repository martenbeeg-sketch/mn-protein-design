# mn-protein-design Missing Work and Improvement Roadmap

Updated: 2026-09-28

This is a conceptual audit. It does not mean the existing application should be
rewritten. The app already contains substantial scientific functionality. The
items below are the highest-value gaps if development continues.

## Current checkpoint

Implemented strengths include:

- normalized candidate sets shared across design, refolding, analysis, and
  benchmarks;
- explicit target/binder chain-role normalization;
- broad generator and refolder coverage;
- multi-engine design campaigns;
- target MSA reuse and evidence-mode controls;
- binder and capacity benchmark workflows;
- queued local-worker execution for several long-running paths;
- runtime estimation and result matrices;
- native output preservation and detailed result views.

The current application source snapshot is committed. Preserve historical run
folders and local source/reference data, but keep simulation outputs and other
user-specific data out of source commits.

## Highest-priority gaps

### 1. First-party automated tests

No top-level first-party `tests/` suite or pytest configuration was found.
`pytest` is not installed in the documented `mn-protein-design` environment.
Tests inside vendored `tools_to_implement/` and `ui_inspiration/` trees do not
validate this application.

Add a CPU/local suite covering:

- job creation and state transitions;
- normalized candidate reading/writing;
- run-relative path resolution;
- explicit and legacy chain roles;
- queue/resource lock behavior;
- candidate import and rank preservation;
- target gap/break analysis;
- workflow parent/child provenance;
- mocked engine output normalization;
- benchmark matrix status precedence;
- runtime estimates;
- Streamlit AppTest coverage for major pages and result routes.

Docker, GPUs, checkpoints, external databases, and networks should remain
separate integration tests.

### 2. Packaging and documentation consistency

`pyproject.toml` now points to the current `README.md`. Some older Markdown
references to the removed `APP_BUILD_PLAN.md` may still need cleanup or clear
historical labeling.

Keep the README and handoff set authoritative, update remaining stale links, and
keep historical notes clearly labeled.

### 3. Runtime portability

The default app home and temporary directory are inside the checkout. Several
workflow constants and Docker mounts still use `/mnt/db/reference_files`
directly, and Settings explicitly lists partially migrated areas.

Future improvements:

- persisted runtime configuration with clear precedence;
- configurable app home, runs, temporary files, references, caches, and datasets;
- no source-checkout mount requirement for installed operation;
- compatibility resolution for existing absolute paths;
- no bulk rewrite of historical runs.

### 4. Typed artifacts and lineage

The current `Artifact` model is small and permits an absolute path when a file is
outside the run directory. Candidate records also retain some source-run absolute
paths for compatibility.

Move incrementally toward:

- artifact type and schema version;
- producer run ID and run-relative path;
- stable artifact ID, size, checksum, role, and media type;
- explicit workflow and upstream/downstream references;
- compatibility loaders for historical candidate paths;
- guarded deletion based on actual consumers.

The normalized candidate contract should remain; typed artifacts should
strengthen it rather than replace its scientific content.

### 5. Durable worker and resource scheduling

The current app uses `worker_request.json`, spawned local worker processes,
heuristic GPU classification, and filesystem resource locks. This supports
background work but is less robust than a supervised durable worker model.

Possible improvements:

- supervised workers with stable identities and heartbeats;
- atomic claims and per-GPU leases;
- CPU, RAM, VRAM, scratch, and exclusivity admission;
- structured waiting reasons;
- exact container CID tracking and cleanup;
- consistent cancellation and immutable retry;
- workflow-child replacement semantics;
- migration of remaining synchronous Streamlit execution.

### 6. Module size and ownership

Several modules are exceptionally large:

- `app/pages/results.py`: about 17,700 lines;
- `workflows/benchmark.py`: about 8,600 lines;
- `workflows/refolding.py`: about 5,900 lines;
- `workflows/design_campaigns.py`: about 4,500 lines;
- multiple Streamlit pages exceed 2,500 lines.

Split only along stable contracts, for example:

- engine adapters;
- input staging;
- candidate normalization;
- metric calculation;
- benchmark aggregation;
- result renderers by job family;
- shared tables/viewers/actions.

Avoid a cosmetic rewrite that duplicates workflow behavior.

### 7. Authoritative engine registry and diagnostics

Engine metadata is distributed across constants, `core/modules.py`, Docker
Compose, Settings, and documentation. The documented module registry does not
cover every newer engine.

A future registry should include:

- engine/tool ID and display aliases;
- image tag and digest;
- accepted and produced artifacts;
- resource requests;
- reference/model requirements;
- code/model/data licenses;
- supported evidence modes;
- health check and smoke fixture;
- implemented, experimental, compatibility, or disabled status.

Add a read-only installation doctor before exposing editable service controls.

## Scientific improvements

### Target import and repair

Current analysis detects numbering gaps and coordinate breaks and supports
derived fragments. Still missing or incomplete:

- authoritative SEQRES/mmCIF versus coordinate-sequence comparison;
- structured missing-segment and modified-residue reports;
- explicit standard-residue mapping provenance;
- optional modeled repair copies;
- validation fixtures for multichain, modified, fragmented, and incomplete
  targets.

Detection and reporting should precede automatic rebuilding.

### Benchmark validity

The app has rich metrics, but future validation should ensure:

- exact candidate coverage for every engine;
- no stale engine-prefixed feature leakage during merges;
- correct binder-versus-full-target chain groups;
- PAE residue ordering remains consistent with structure normalization;
- completed prediction is distinct from completed metric evaluation;
- uncertainty and replicate behavior are visible where applicable;
- benchmark presets are recorded as actual engine evidence, not only UI labels.

### Campaign robustness

Continue hardening:

- resume at explicit stage/candidate boundaries;
- partial failure without losing successful engine outputs;
- immutable campaign inputs and child provenance;
- exact level-1/level-2 settings separation;
- stable source and normalized structure references;
- reproducible seeds and workload summaries.

## What can be borrowed from mn-ligand

Useful infrastructure concepts:

- run-relative `ArtifactRef` and manifests;
- explicit tool manifests with resource and license metadata;
- durable worker heartbeats and per-GPU leases;
- resource admission snapshots and waiting reasons;
- shared cancellation/retry controls;
- common result tabs and typed downstream actions;
- runtime-path configuration and installation diagnostics;
- preserving historical runs through loaders rather than rewrites.

Do not borrow ligand-specific workflows, compound identity rules, docking score
semantics, MD/free-energy contracts, or ligand-page navigation.

## Proposed phases

### Phase 0: Honest baseline

- establish these handoff documents;
- repair the package README reference;
- inventory active engines and synchronous versus queued paths;
- create the first first-party pytest/AppTest baseline.

### Phase 1: Contract hardening

- typed run-relative artifacts;
- candidate compatibility resolver;
- authoritative engine registry;
- installation diagnostics.

### Phase 2: Worker consolidation

- supervised durable worker;
- resource admission and heartbeats;
- consistent cancellation/retry;
- migrate remaining page-owned subprocesses.

### Phase 3: Modularization

- extract engine adapters and result renderers;
- reduce monolithic benchmark/refolding/result modules;
- retain behavior through characterization tests.

### Phase 4: Scientific maturation

- target import/repair reports;
- benchmark fixtures and metric validation;
- campaign resume and partial-failure hardening;
- reproducible image/reference qualification.

## Validation gates for an engine

An engine should be called implemented only when:

1. source/version, image, licenses, references, and citation are recorded;
2. a minimal native run produces non-empty outputs;
3. failures are actionable;
4. outputs normalize into valid candidates with explicit chain roles;
5. native and normalized evidence are visible in Results;
6. a downstream workflow consumes the candidates;
7. worker cancellation and resource cleanup are verified;
8. focused local regression tests protect staging and normalization;
9. GPU/driver compatibility is tested on the intended host;
10. benchmark completion includes the requested metric layers, not prediction
    files alone.
