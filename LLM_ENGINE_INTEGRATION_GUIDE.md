# LLM Engine Integration Guide

This file is for future LLM/coding agents adding, repairing, or benchmarking a
folding/refolding engine in `mn-protein-design`.

Read it chronologically. The safest order is:

1. Install/containerize the engine.
2. Prove the engine runs outside the app.
3. Implement the app wrapper in the refolding workflow.
4. Normalize outputs to the app candidate contract.
5. Wire benchmark orchestration.
6. Wire UI/result/matrix/runtime surfaces.
7. Test through the same worker path the user will use.

Do not start by changing benchmark staging. Existing benchmark runs already work;
new engines should adapt to the app's established benchmark contract.

## 0. First Orient Yourself

Before editing, inspect the current code because constants and defaults drift:

```bash
rg -n "ENGINE_NAME|PROTENIX|OPENFOLD|BOLTZ|RF3" mn_protein_design containers docker-compose.yml
rg -n "run_.*complex_refolding|run_de_novo_binder_scoring_dataset" mn_protein_design/workflows
```

Useful files:

* `APP_FEATURES.md`
* `RUN_APP.md`
* `mn_protein_design/workflows/refolding.py`
* `mn_protein_design/workflows/benchmark.py`
* `mn_protein_design/app/pages/benchmark.py`
* `mn_protein_design/core/jobs.py`
* `mn_protein_design/core/local_worker.py`
* `mn_protein_design/core/runtime_estimator.py`

Every implementation claim in this guide should be verified against current
code before acting on it.

## 1. Install Or Containerize The Engine

Start with the engine itself, not the benchmark UI.

Add or update:

* `containers/<engine>/Dockerfile`
* `docker-compose.yml` service when useful
* image name constants in workflow code
* reference/model path constants in workflow code

Large model files, checkpoints, weights, MSA databases, and tool caches belong
under `/mnt/db/reference_files`, not in the repo, not in run artifacts, and not
in `/root` or `/home/user`.

Typical reference layout:

```text
/mnt/db/reference_files/<engine>/
```

Example current Protenix CLI layout:

```text
host:       /mnt/db/reference_files/protenix
container:  /ref/protenix
env:        PROTENIX_ROOT_DIR=/ref/protenix
image:      mnprot-protenix-cu128:latest
dockerfile: containers/protenix/Dockerfile
```

Use stable mount paths. If an upstream downloader defaults to a user home or
temporary cache, redirect it explicitly.

## 2. Prove The Engine Works Standalone

Before touching app logic, run a minimal engine prediction outside the app:

1. Confirm the Docker image builds.
2. Confirm model weights are visible in the mounted reference directory.
3. Run one tiny input.
4. Confirm output files exist.
5. Confirm GPU visibility and memory behavior.
6. Record the exact command shape in code comments or this guide if it is
   unusual.

Do not proceed to benchmark wiring until one standalone prediction has worked.

For Protenix CLI, verify the current command shape in
`run_protenix_cli_complex_refolding`. At the time this guide was last updated,
the command used:

```text
docker run --rm --gpus ... --shm-size=32G
  -v <run_dir>:/work
  -v /mnt/db/reference_files/protenix:/ref/protenix:rw
  -w /work
  -e PROTENIX_ROOT_DIR=/ref/protenix
  -e CUTLASS_PATH=/opt/cutlass
  mnprot-protenix-cu128:latest
  protenix pred
    -i /work/<input_dir_or_batch_dir>
    -o /work/<output_dir>
    -s <seeds>
    -n <model_name>
    -c <cycles>
    -p <diffusion_steps>
    -e <samples>
    -d bf16
    --use_msa <bool>
    --use_template <bool>
    --use_default_params <bool>
    --need_atom_confidence True
```

## 3. Implement The Refolding Wrapper First

If the engine folds/refolds candidates, implement it first in
`mn_protein_design/workflows/refolding.py`.

The wrapper should:

1. Accept `source_run_dir` and `candidates_jsonl`.
2. Read candidates with the existing candidate helpers.
3. Stage native inputs under `artifacts/raw/<engine>/`.
4. Write a clear `command.json`.
5. Launch Docker or local commands.
6. Verify output completeness.
7. Convert outputs to normalized candidates.
8. Write `artifacts/normalized_candidates/candidates.jsonl`.
9. Call `finish_job`.

Keep one implementation reusable from:

* Refolding / Validation
* Refolding after designs
* Target refolding, when applicable
* Binder Benchmark
* Future workflows that consume normalized candidates

If the engine is a generator rather than a refolder, implement the native
generator path in `workflows/design.py` or `workflows/design_campaigns.py`, then
normalize outputs. Do not make benchmark code read a generator's private output
layout directly.

## 4. Respect The Job Contract

Every run should be inspectable from:

```text
mn-protein-design-workdir/workdir/runs/<task-group>/<run_id>/
  input.json
  metadata.json
  command.json
  stdout.log
  stderr.log
  result.json
  artifacts/
```

Use helpers from `mn_protein_design.core.jobs`:

* `create_job`
* `update_status`
* `finish_job`
* `stop_job`
* `pause_job`
* `resume_job`
* `write_json`
* `read_json`

Do not hand-edit metadata or result files in normal code paths. These helpers
keep queue locks, timestamps, result state, and UI state consistent.

## 5. Normalize Outputs

The app's durable exchange format is:

```text
artifacts/normalized_candidates/candidates.jsonl
```

Each successful engine prediction should become a normalized candidate with:

* `candidate_id`
* `stage`
* `tool` / `source_tool`
* `complex_pdb` or equivalent structure path
* `binder_chains`
* `target_chains`
* `parents`
* `metrics`
* `raw_metadata`

### Chain Role Convention

New engine integrations must keep biological roles explicit. Do not infer
target/binder identity from alphabetical order alone.

Use the explicit chain-role staging convention:

* Target/reference/scaffold chains are staged forward from `A`, `B`, `C`, ...
* Binder/design chains are staged backward from `Z`, `Y`, `X`, ...
* Chain maps should include `chain_role_schema` and `chain_roles` metadata.
* Keep legacy-compatible `binder` and `targets` chain-map sections until all
  consumers have moved to `chain_roles`.

The helper module is `mn_protein_design/workflows/chain_roles.py`. Use it
instead of hand-picking chain IDs:

```python
target_engine_chains = chain_roles.assign_target_engine_chains(target_chains)
binder_engine_chains = chain_roles.assign_binder_engine_chains(
    binder_chains,
    reserved=target_engine_chains,
)
```

For target-only refolding, there is no binder role: fragments/domains remain
target chains and should be staged as `A`, `B`, `C`, ... . For binder-target
benchmark/refolding, the binder must not be silently converted back to `A` in
normalized candidates, viewer metadata, PAE/ipSAE scoring inputs, or Rosetta
evaluation inputs.

Validate role metadata before handing structures to engines or scoring:

```python
warnings = chain_roles.validate_chain_roles(
    candidate_id=candidate_id,
    binder_chains=binder_engine_chains,
    target_chains=target_engine_chains,
    structure_chains=structure_chains,
    schema=chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
)
```

Treat validation output as non-blocking findings unless the input cannot be
staged at all. Legacy data without `chain_role_schema` is allowed and should be
classified as `legacy_inferred` / compatibility information, not as a problem.
Only warning/error findings such as binder-target chain overlap or declared
chains missing from the structure should increment warning counts.

Engines and design tools may keep their native chain behavior internally. The
app should normalize at workflow boundaries instead of forcing every tool to
adopt the app chain convention. Use `normalize_candidate_structure_roles(...)`
after a prediction when a tool emits a native/legacy structure. The normalized
candidate should point to the app-normalized structure while raw metadata keeps
the original prediction path.

### Staged Design Campaign Chain Roles

The staged Design Campaign path is a high-risk integration surface because it
chains generator outputs into sequence design, internal refolding screens, final
refolding, and metrics.

Rules for staged generator campaigns:

* Treat the selected target PDB and selected target chains as the authority for
  target identity.
* After every generator and sequence-design step, re-read the produced structure
  and reconcile `binder_sequence`, `binder_length`, `binder_chains`,
  `target_chains`, `chain_role_schema`, `chain_roles`, and `raw_metadata`.
* Do not trust an old JSON `binder_sequence` if it conflicts with the declared
  binder chain in the structure. Recover from the actual staged binder chain
  where possible and record the source path in metadata.
* Sequence-native generators such as ESMFold2 binder design and
  Proteina-Complexa may emit sequence+structure candidates directly. They can
  skip a level-1 MPNN redesign, but still need the same role metadata before
  internal refolding, contact/hotspot metrics, and final evaluation.
* Engine-scout / pilot workflows should use the same generator normalization
  path as full staged campaigns. Do not maintain a separate "scout-only" parser
  that drifts from the real campaign path.
* If an engine needs a legacy local chain layout, create that layout only in the
  engine-specific staging folder and normalize back to the app role contract at
  the workflow boundary.
* After each generator emits a structure, calculate the same early structural
  metrics used by scout plots and staged filters: hotspot/contact fraction where
  hotspots exist, binder radius of gyration, target-aligned binder RMSD where a
  target-aligned comparison is meaningful, PyDSSP secondary-structure fractions
  and element counts, and CA-trace geometry element counts.
* Do not wait until final evaluation to calculate these generation-stage metrics.
  Staged level-1/level-2 sequence-redesign filters can use them before internal
  refolding.
* Keep optional pre-redesign structure filters explicit in the campaign input.
  A common current filter is minimum secondary-structure element count greater
  than or equal to 3, with PyDSSP counts used for normal backbone structures and
  CA-trace geometry counts used for Genie3-style reconstructed backbones.
* Sequence redesign defaults should avoid cysteine and unknown residues through
  the omitted amino-acid list `C,X` unless the user deliberately changes it.
  Ensure only the level-specific redesign settings are consumed for that level;
  stale global or older-level sequence-design parameters should not leak across
  levels.

AF2 initial guess is the current known sharp edge. The app-level convention is
target `A...` and binder `Z...`, but AF2-IG historical code assumed the binder
was `A` or `B`. When using AF2-IG as an internal staged screen, derive the
designed/binder chain from staged sidecars such as
`.binder_source_chains.txt`/`.chain_map.json`, or use an AF2-IG-local
compatibility input. Do not silently pass `--designed_chains A` for a staged
complex whose binder is `Z`.

Benchmark code should consume normalized candidates and summary metrics, not
engine-private folders.

Keep internal engine keys stable even when display names change. For example,
legacy PXDesign-backed Protenix still uses internal keys, folders, and metric
prefixes such as `protenix`, `protenix_metrics.csv`, and `protenix_*`, but the
UI/result label is `Protenix v0.5`. Use explicit alias/canonical-label helpers
instead of renaming historical files or prefixes.

When detecting feature ownership from metric prefixes, match the most specific
prefix first. For example, `protenix_v1_` and `protenix_v2_` must be checked
before `protenix_`, otherwise v1/v2 features will be swallowed by the legacy
engine bucket.

### Genie3 Backbone Reconstruction

Genie3 and similar generators may emit CA-only or otherwise incomplete backbone
representations. For staged sequence design, viewer display, and secondary
structure analysis, reconstruct a no-relaxation glycine backbone from the CA
trace when full backbone atoms are missing. This is intended to match the way
RFdiffusion-style glycine backbones can be handed to ProteinMPNN-compatible
steps.

Do not relax the reconstructed structure unless the workflow explicitly requests
relaxation. Record both the native source path and the reconstructed glycine
backbone path in candidate metadata so the viewer and metric code can choose the
right artifact.

PyDSSP may classify reconstructed or unusual backbones as mostly coil. Keep
CA-trace geometry element counts alongside PyDSSP counts so filters and plots can
distinguish genuinely unstructured outputs from CA-trace reconstructions whose
regularity is visible geometrically but not accepted by DSSP-style hydrogen-bond
criteria.

## 6. Add Benchmark Wiring

Only after the refolding wrapper works, wire the engine into
`run_de_novo_binder_scoring_dataset` in `workflows/benchmark.py`.

Benchmark integration checklist:

1. Add function parameters and worker payload entries.
2. Add engine label/order/settings helpers.
3. Launch the engine as a hidden child job.
4. Pass the benchmark candidate set derived from the app-produced `run.csv`.
5. Summarize the child with `_summarize_child_candidate_metrics`.
6. Add the engine metrics table to `metric_csvs`.
7. Add runtime timing with `_record_runtime_timing`.
8. Let shared benchmark postprocessing calculate ipSAE/common interface,
   Rosetta, and PyMOL metrics.
9. Resume only when a completed metrics table has expected row coverage.

The benchmark pipeline already prepares:

```text
artifacts/raw/de_novo_binder_scoring/output/run.csv
```

Treat that processed `run.csv` and the normalized benchmark candidate set as the
benchmark contract. Do not redesign benchmark input staging to fix one engine.

Do not let shared input/reference metrics leak into engine identity. Input
PyRosetta/Rosetta tables can be reused across engines, but they must not make a
Protenix-only run appear to contain AF3, Boltz-2, ColabFold, or other engines.
When merging metric tables, drop stale engine-prefixed columns from source CSVs
before adding the metric sidecars for the current run.

Collections should merge source artifacts per engine instead of letting the
last source overwrite earlier source artifacts. This matters when a collection
combines a base benchmark with partial engine runs, for example a 12-target
Protenix run plus a single-target Protenix repair run.

## 7. Add UI Controls

Add engine controls in `mn_protein_design/app/pages/benchmark.py` and any
relevant refolding/design pages.

UI surfaces to check:

* Binder Benchmark engine controls
* Refolding / Validation controls
* Design Campaign final evaluation controls, if the engine is available there
* Results page discovery of metrics and structures
* Runtime estimate tables
* Matrix settings labels

Settings labels matter. Matrix rows with different models or parameters should
be distinguishable.

Engine colors should be stable and visually distinct. After adding a new engine
or splitting one engine into variants, update the results color domain/range so
legends remain readable. Do not let adjacent variants share nearly identical
gray or pale colors.

### Evidence Presets And Refolding Modes

Target Refolding and Binder Benchmark should expose the same bulk evidence
presets wherever the engine set overlaps:

* `Template + MSA engines`
* `Template-only engines`
* `MSA-only engines`
* `No MSA + no template`

When changing one page, check the other page in the same pass. The button names
are user-facing workflow contracts, not cosmetic shortcuts.

The current interpretation is:

* Template + MSA selects engines that can consume both a target structure/template
  signal and a target-chain MSA signal.
* Template-only selects template-capable engines and disables target MSAs.
* MSA-only selects MSA-capable engines and disables target templates.
* No MSA + no template disables every template/initial-guess and target-MSA path
  and should also disable AF2-IG, because AF2-IG has no true sequence-only mode
  in this app.

ESMFold2 belongs in both the template-capable and MSA-capable preset families:
its `initial_guess` mode is the template-like/distogram-conditioning path, and
its target-MSA option passes prepared per-target-chain A3M files into the
ESMFold2 `ProteinInput` objects. Therefore:

```text
Template + MSA  -> modes=["initial_guess"], use_target_msa=True
Template-only   -> modes=["initial_guess"], use_target_msa=False
MSA-only        -> modes=["sequence"],       use_target_msa=True
No evidence     -> modes=["sequence"],       use_target_msa=False
```

Do not infer evidence mode only from the parent run label. Inspect the child
engine input JSON/manifest and command artifacts when debugging whether an
engine actually consumed templates, MSAs, both, or neither.

### MSA Staging And Sanitization

Target MSAs are cached in a shared sequence-hashed A3M repository. Engines should
consume those files only when the corresponding per-engine `use_target_msa`
setting is enabled.

Before handing cached A3M files to strict parsers, sanitize them to
match-column-only rows so every aligned sequence has the same length. Protenix
v0.5 and ESMFold2 have both exposed failures from raw A3M insert columns. The
sanitized copy should be staged into the run artifacts or a temporary file, not
written over the shared cache.

For RF3, query-only fallback MSAs are acceptable when a real target MSA is not
requested or not available. For ColabFold, paired/target MSA handling can be
important for fragmented/multichain targets; validate the actual input files
before assuming a failed structure is a model limitation.

Target Preparation's analysis page can show MSA depth and coverage PNGs for
full-chain and fragment-level MSAs. If fragment-aware target metadata exists,
keep fragment rows visible; older benchmark targets imported before
fragment-awareness may only have coarse full-chain rows.

## 8. Hidden Child Jobs

Benchmark engine predictions should normally be hidden child jobs. Mark them
with:

* `parent_run_dir`
* `parent_run_id`
* `parent_task_group`
* `parent_role="benchmark_engine_subrun"`
* `benchmark_engine=<engine_key>`

The matrix, result browsing, and recovery logic depend on these fields.

If a child runs inside a parent that already owns the GPU lock, do not make the
child compete for the same lock. Use the existing shared-parent metadata pattern
so the UI can show the relationship.

## 9. GPU Queue Rules

Use the existing queue system in `core/jobs.py`.

Important rules:

* A parent benchmark can own `gpu:1` while a hidden child performs prediction.
* A child sharing the parent's lock should not block behind that parent.
* Queue owner detection must ignore the requester run itself.
* If a job stays queued, check active owners and stale hidden children before
  assuming Docker failed.
* If a stale job is blocking the queue, use `stop_job`; do not edit metadata by
  hand.

If starting workers manually from an LLM/tool environment, make sure the process
can write to `/mnt/data/RESULTS/...`. A sandboxed worker that can only read the
workdir will fail or silently not progress like the real app worker.

## 10. Resume Rules

Resume is useful but dangerous.

Safe resume requires:

* The completed engine metrics table exists.
* The table has expected row coverage.
* The child candidate IDs match the current benchmark candidate set.
* Engine settings are compatible with the current request.
* The current run is not inheriting a stale hidden child from another run.

Do not reuse an incomplete child just because the engine name matches.

For new engines, implement resume conservatively. It is better to rerun a small
amount of work than to fold the wrong candidate set for hours.

## 11. Batch Mode Rules

Batching is optional. Implement it only after single-input execution works.

Batch mode is useful when an engine pays large Docker/model setup overhead.
However, batching must preserve observability and resume behavior.

For benchmark engines, batches should be target-aware when possible:

* Avoid mixing unrelated targets in one batch if the matrix is expected to show
  target-level progress.
* Store `current_candidate_ids` in child metadata.
* Store `current_target_ids` in child metadata.
* Store `current_step_index` and `current_step_total`.
* Write `artifacts/runtime_step_timings.json`.
* Include target IDs, residue totals, start/end timestamps, elapsed seconds,
  candidate count, and return code per batch.

Batch size tradeoff:

* Larger batches reduce repeated Docker/model setup.
* Larger batches make resume coarser because a failed batch retries more
  entries.
* If the engine processes inputs serially inside one container, GPU memory is
  often dominated by the largest complex, not the number of inputs. Verify this
  from logs and GPU memory, do not assume it.

Current Protenix CLI code should be verified before use. At the time this guide
was last updated, `run_protenix_cli_complex_refolding` had a default
`batch_size` of `32`, but an agent should always confirm this in
`workflows/refolding.py`.

## 12. Matrix Rules

The benchmark matrix should answer:

* Which engine-target cells are completed?
* Which are queued?
* Which exact target is running?
* Which failed?
* Which are missing from current coverage?

Do not let the matrix guess running targets. If active child IDs or target IDs
cannot be mapped to a target, keep target cells queued and show child progress
in the live progress table instead.

Status priority for live master coverage:

```text
completed > running > queued > failed > missing
```

Failed newer attempts should not hide older completed benchmark cells.

## 13. Runtime Estimates

Record enough data to estimate future runs:

* engine key
* candidate count
* total residues
* max system residues when available
* elapsed seconds
* batch timing rows where applicable

For batched engines, show both:

* residue-based estimate
* batch-based estimate

When fixed setup overhead is large, use a conservative combined estimate.
Residue-only estimates can be much too optimistic for engines that repeatedly
start Docker/model inference.

## 14. Metrics Contract

Prediction files are not enough for a completed benchmark.

A benchmark engine is complete only after metrics are calculated and merged.
Expected downstream metric families:

* ipSAE/common interface metrics
* Rosetta metrics for predicted complexes
* PyMOL metrics
* input PyRosetta metrics when selected

If input PyRosetta metrics are invariant for the benchmark dataset, reuse cached
input metrics when complete coverage exists. Do not recompute them separately
for every engine if the input structures are identical.

### Chain Contract For Metrics

For structure-based benchmark metrics, the intended interpretation is:

```text
binder group vs whole target group
```

Rosetta and PyMOL metrics are staged toward a two-group contract:

```text
binder chain(s) -> A
all declared target chain(s) -> B
```

Rosetta then scores `A_B` with `InterfaceAnalyzerMover`, and PyMOL evaluates
chain `A` against chain `B`. This is the right benchmark interpretation for
multichain targets: one binder against the full target assembly.

ipSAE/common interface metrics are also intended as binder against all target
chains. The current batch aggregation handles multichain native layouts by
calculating binder-vs-each-target-chain pairs and aggregating across target
partners. Therefore, for a target with chains `B,C,D`, `ipSAE_min` is an
aggregate over `A-B`, `A-C`, and `A-D`, not a random single target chain.

Do not casually rewrite ipSAE staging to collapse chains unless you also prove
the PAE matrix still matches the staged residue order. Renaming or reordering
chains can silently corrupt PAE-indexed metrics. The current behavior is
acceptable and should be documented rather than changed without a concrete
bug.

## 15. Protenix-Specific Notes

There are two Protenix-related paths:

* legacy PXDesign-backed `Protenix v0.5` in UI/results, internally `protenix`
* standalone Protenix CLI v1/v2, currently represented by `Protenix v1` and
  `Protenix v2`

Keep these paths separate unless intentionally refactoring them together.

Current standalone Protenix items to verify in code:

* image constant: `PROTENIX_CLI_IMAGE`
* reference constant: `PROTENIX_CLI_REFERENCE_DIR`
* v1 model constants: `PROTENIX_V1_MODEL`,
  `PROTENIX_V1_20250630_MODEL`
* v2 model constant: `PROTENIX_V2_MODEL`
* refolding entry point: `run_protenix_cli_complex_refolding(...)`
* benchmark flags: `run_protenix_v1`, `run_protenix_v2`
* metrics tables: `protenix_v1_metrics.csv`, `protenix_v2_metrics.csv`
* result structure folders: `predicted_metric_pdbs/protenix_v1`,
  `predicted_metric_pdbs/protenix_v2`,
  `predicted_viewer_structures/protenix_v1`,
  `predicted_viewer_structures/protenix_v2`

At the time this guide was last updated, current code used:

```text
PROTENIX_CLI_IMAGE = "mnprot-protenix-cu128:latest"
PROTENIX_CLI_REFERENCE_DIR = /mnt/db/reference_files/protenix
PROTENIX_V1_MODEL = "protenix_base_default_v1.0.0"
PROTENIX_V1_20250630_MODEL = "protenix_base_20250630_v1.0.0"
PROTENIX_V2_MODEL = "protenix-v2"
```

Do not rely on this paragraph without checking `workflows/refolding.py`.

Protenix lessons:

* Keep benchmark staging unchanged; adapt Protenix to the benchmark workflow.
* Use the app-produced benchmark `run.csv` and normalized candidate set.
* Validate weights and `/mnt/db/reference_files/protenix` before long runs.
* Hugging Face may be used to obtain weights when the upstream downloader fails,
  but the final app path should remain stable under `/mnt/db/reference_files`.
* Protenix ranking scores are engine-native confidence/ranking values, not
  benchmark success. Benchmark metrics still need to run.
* Keep `Protenix v0.5`, `Protenix v1`, and `Protenix v2` distinct in results,
  matrix rows, feature ownership, color scales, and PAE x Rosetta combination
  features.
* Do not rename legacy `protenix_*` columns to `protenix_v0_5_*`; preserve old
  artifacts and map only the display label.
* Protenix CLI template mode is not the same thing as an AF2-IG/Boltz-2 rigid
  input-template layout. Protenix reads `templatesPath` JSON files per
  `proteinChain`; its template featurizer recenters template coordinates. If a
  broken target chain was split into fragment chains and each fragment is passed
  as its own Protenix `proteinChain`, Protenix can accept all templates while
  losing the original relative fragment geometry. Logs can therefore look
  correct (`Found 1 templates...`, `N_asym` equals fragment count) even when the
  structural result is bad.
* For target-only split-fragment inputs, Protenix should receive one collapsed
  target `proteinChain` plus one `templatesPath` JSON that embeds all fragment
  coordinates as a single-chain mmCIF. This preserves the relative positions
  inside one template coordinate frame. Keep the split-fragment multi-chain
  representation for engines that support it directly, but collapse it at the
  Protenix JSON boundary.
* Because the collapsed Protenix representation is still a connected sequence,
  Protenix-family engines are not the default choice for fragmented targets.
  Target-refolding UI presets should prefer AF2 template and Boltz-2 for these
  rows, leaving Protenix available only as an explicit/manual diagnostic choice.
* The Protenix collapsed-fragment path should record manifest fields such as
  `template_chain_mode="collapsed_fragments"`, `source_template_chains`, and an
  exact `template_aligned_residues` count. If a run still shows one Protenix
  chain/template per fragment for a target-only split-fragment case, it is using
  the old/bad staging mode and should be rerun.
* Do not attach stale per-fragment target MSAs to a collapsed Protenix target
  sequence. Unless a real MSA exists for the exact collapsed sequence, prefer
  template-only Protenix input for this repair/refolding mode and record the MSA
  status explicitly in the manifest.

## 16. Anti-Patterns

Avoid these:

* Creating a new benchmark input mode to fix an engine-specific problem.
* Comparing an external shard CSV directly to the processed benchmark `run.csv`
  and assuming mismatch means the app is wrong.
* Starting a long benchmark before a one-input smoke test passes.
* Marking a matrix cell completed because a child job exists.
* Marking all targets in an engine row running when only one target/batch is
  active.
* Reusing an old child job because the engine name matches.
* Leaving a hidden child marked running after its Docker container was stopped.
* Starting workers from a restricted environment that cannot write to the
  workdir.
* Treating prediction completion as benchmark completion before metric
  postprocessing.
* Assuming AF2 initial guess still sees the binder as chain `A` after staged
  campaign normalization. In the current app role contract, binder chains are
  normally `Z`, `Y`, `X`, ... .
* Assuming a Protenix run used target templates correctly just because
  `--use_template True` appears in the command. Inspect the generated Protenix
  JSON and `msa_manifest.json`; target-only split-fragment inputs should be one
  collapsed protein chain with one fragment template, not one template per
  fragment chain.
* Inferring engine coverage from any engine-prefixed column in a merged
  collection without checking the selected run parameters and engine metric
  sidecar existence.
* Letting collection-local artifact folders be symlinks to only one partial
  source when the collection is supposed to combine multiple partial sources.
* Treating an engine display rename as a storage migration.

## 17. Minimal Verification Before Handing Back

After implementing or changing an engine:

1. Compile touched Python files:

   ```bash
   python3 -m py_compile <touched files>
   ```

2. Build or verify the Docker image.
3. Run one standalone engine smoke test.
4. Run one tiny app/refolding job through the real worker path.
5. Confirm normalized candidates are written.
6. Run a tiny benchmark job through the real worker path.
7. Confirm the parent reaches the expected engine phase.
8. Confirm hidden child metadata has the expected engine key.
9. Confirm Docker logs show expected model, weights/cache path, and candidate or
   batch size.
10. Confirm prediction artifacts become benchmark metrics tables.
11. Confirm ipSAE/common interface, Rosetta, and PyMOL metrics run when selected.
12. Confirm the matrix shows completed/running/queued/failed cells truthfully.

If a long benchmark is already running, inspect metadata and logs before
stopping it. Stop only when the run is demonstrably folding the wrong candidates
or blocking the queue.
