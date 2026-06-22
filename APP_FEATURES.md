# mn-protein-design Available Features

This document describes the features currently exposed by the app and wired in
the codebase. `APP_BUILD_PLAN.md` remains the longer behavior contract and
roadmap; this file is the shorter current-state inventory.

## App Shape

`mn-protein-design` is a local Streamlit workbench for Docker-orchestrated
protein binder design, validation, analysis, and binder/nonbinder benchmarking.

The app is organized around file-backed jobs. Each run writes:

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

The main navigation includes:

* Jobs
* Target Preparation
* PPI / Hotspot Detection
* Target Cropping
* Design
* Sequence Design
* Candidate Sets
* Refolding / Validation
* Analysis
* Binder Benchmark
* Capacity Benchmark

## Job And Runtime Features

Implemented runtime behavior includes:

* file-backed run store under `mn-protein-design-workdir`
* stable contract files for inputs, metadata, command, logs, results, and artifacts
* normalized candidate sets in `artifacts/normalized_candidates/candidates.jsonl`
* shared GPU/resource queue for GPU-labelled tools
* hidden child-job mechanics for multi-engine benchmark/refolding execution
* on-demand result ZIP generation from result pages
* deletion planning with optional downstream-job detection
* job codes for compact UI references
* runtime estimates for benchmark/refolding engines from completed run history

## Target Preparation

The target workflow supports:

* uploaded local PDB/mmCIF-style structures
* target chain selection
* chain-aware trimming/preparation
* noncanonical residue reporting/handling hooks
* PDB numbering and chain mapping artifacts
* prepared target artifacts that downstream design and detection pages consume

Typical outputs include target PDB, target metadata, residue numbering maps, and
viewer state files.

## PPI / Hotspot Detection

The PPI / Hotspot Detection page exposes:

* ScanNet PPI/interface scoring
* Surf2Spot hotspot prediction
* MaSIF-seed target-surface support
* prepared-structure preview
* existing detection result browsing

Detection jobs write native tool artifacts plus normalized outputs where the
workflow supports them.

## Target Cropping

Target Cropping is separate from target preparation. The page provides:

* Mol* structure/sequence selection support
* manual residue selection
* sphere-style selection expansion
* plane/slice controls
* crop draft preview
* confirmed crop output suitable for downstream design

## Design

The Design page is the main binder-design entry point. Shared controls include:

* prepared or cropped target selection
* target chain selection
* hotspot/epitope selection from text or Mol*
* binder length/range
* design attempts
* campaign name
* per-tool payload preview before launch

Available design tabs:

* RFdiffusion classic
* BindCraft
* RFdiffusion3 / Foundry
* BoltzGen
* PXDesign
* Genie3
* ESMFold2 native binder design and experimental sequence screening
* Protpardelle-1c
* Proteina-Complexa

### Vanilla / Native Design Modes

Several engines preserve a native or vanilla-style workflow instead of forcing
all tools into one generic pipeline.

Available native/vanilla design integrations:

* **BindCraft**: native end-to-end pipeline with BindCraft settings, filters,
  accepted designs, and native tables.
* **BoltzGen**: native BoltzGen workflow with ranked native designs and metrics.
* **PXDesign**: native PXDesign output and summary handling.
* **Genie3**: native workflow with selectable ColabFold or Boltz2-style backend
  settings and model/reference cache paths.
* **Proteina-Complexa**: native binder pipeline with native CSV parsing and
  ranking from native confidence where available.
* **Protpardelle-1c**: target-motif-conditioned workflow that writes a compact
  motif PDB, remaps hotspots, runs Protpardelle sampling, and normalizes
  produced structures/ESMFold metrics.
* **ESMFold2 native binder design**: Biohub's released gradient-guided
  ESMFold2/ESMC sequence-design protocol, run in the Biohub ESM container with
  a lean shared-model profile and normalized complex candidates.

Available app-orchestrated design pipelines:

* **RFdiffusion classic**: generation-only or full vanilla-style app pipeline.
  The full path can run generation, sequence design, monomer refolding, complex
  refolding, and analysis.
* **RFdiffusion3 / Foundry**: RF3/Foundry generation with Foundry-native mapping
  and downstream normalized candidates.
* **ESMFold2 experimental screening**: screens candidate binder sequences or
  structures through ESMFold2-style complex prediction/scoring.

Every design backend that produces candidates should emit:

```text
artifacts/normalized_candidates/candidates.jsonl
artifacts/normalized_candidates/campaign_result.json
```

Native output tables and structures are preserved under `artifacts/raw/<tool>/`
where possible.

## Sequence Design / Optimization

The Sequence Design page consumes normalized backbone candidates and supports:

* shared LigandMPNN container for PDB candidates
* ProteinMPNN weights through the LigandMPNN container
* LigandMPNN
* SolubleMPNN
* Foundry-native MPNN for Foundry/RF3 CIF-style candidates
* configurable sequences per backbone, sampling temperature, omitted amino acids,
  design chains, seed, batches, and batch size

Optional downstream continuation can run:

* monomer refolding
* complex refolding
* analysis

## Candidate Sets

The Candidate Sets page supports importing external results into the app's
normalized candidate format.

Currently implemented imports:

* generic CSV/TSV/Excel candidate tables with configurable candidate-ID,
  binder-sequence, structure-path, and chain-role columns
* preservation of unmapped source-table columns as `import_*` candidate metrics,
  including source ranking as `import_rank`
* source-aware candidate ordering: generic imports use `import_rank`, while
  BindCraft imports retain `bindcraft_final_rank`
* BindCraft output folder import
* optional target PDB
* binder/target chain role assignment
* glob-based structure selection
* max-candidate limit
* copy or symlink staging into the app workdir

The page also lists all available normalized candidate sets from design,
refolding/validation, analysis, and candidate-import jobs.

## Campaigns

Campaigns string normalized modules together from a starting backbone candidate
set.

Available campaign steps:

* sequence design
* monomer refolding
* complex refolding
* analysis

Campaigns preserve step lineage and create child jobs whose outputs feed the
next selected module.

## Job Operations

The Jobs page provides shared file-backed operations across queued local-worker
tasks:

* selected-row pause, stop/cancel, resume, and delete controls
* pause/stop sends a termination signal to the recorded local-worker process
  group and releases GPU/resource locks
* resume requeues paused/stopped local-worker jobs when `worker_request.json`
  is present, preserving existing artifacts
* runtime estimates for job rows based on completed-run history when available
  and conservative fallback rates otherwise
* GPU queue-resource visibility so independent jobs pinned to different GPUs
  can run concurrently while jobs sharing one GPU wait for the lock

## Refolding / Validation

Refolding / Validation consumes normalized candidates from design, import,
analysis, or previous refolding jobs. It can run as a queued background worker
so navigation can continue while long evaluations run.

Dataset features:

* selectable candidate set
* per-candidate checkbox selection
* select-all behavior
* source-aware candidate sorting controls, including imported table rank and
  native BindCraft rank when available
* optional target structure override
* target library entries from candidate sources and installed benchmark targets

Prediction/refolding engines:

* AlphaFast AF3
* ColabFold / AF2-Multimer
* AF2 target-only initial guess
* ESMFold2
* Boltz-2 target-template prediction/refolding
* RF3
* Protenix
* BoltzGen Fold

Engine settings include:

* GPU device
* shared target MSA source
* MSA repository path
* AlphaFast/MMseqs database path
* requirement for real target MSAs
* per-engine model/cache/checkpoint paths
* recycles, diffusion/sampling steps, samples, seeds, and template toggles
* selected target PDB coordinates plus binder sequence as the default AF2 input
* explicit legacy whole-complex AF2 initial-guess controls
* target-template and full-PAE controls for Boltz-2

Metrics/evaluation options:

* native engine confidence parsing
* recalculated interface scores such as ipSAE, iPAE, LIS, pDockQ, and pDockQ2
* Rosetta metrics for input complexes
* Rosetta metrics for predicted complexes
* PyMOL interface/geometry metrics
* sequence-only imports can run refolding plus predicted-complex metrics without
  input PyRosetta; input PyRosetta is reserved for candidate sets that provide
  an original binder-target complex

Refolding evaluation outputs are stored as benchmark-family jobs with the
`refolding_evaluation` job type and canonical benchmark-style result layout.

## Analysis

The Analysis page separates native pipeline results from app re-analysis.

Native result display is available for:

* BindCraft
* BoltzGen
* PXDesign
* Proteina-Complexa
* Genie3
* Protpardelle-1c
* RFdiffusion3 / Foundry
* RFdiffusion classic

Native result mode shows native tables and native structure viewing, then stops.
It does not automatically mix native design conclusions with app-level
re-analysis.

App re-analysis is available for normalized complex-refolded candidates and
produces:

* ranked candidate table
* pass/fail counts
* metrics summary
* normalized analysis-stage candidates
* optional IPSAE/iPAE-derived metrics when PAE data is available
* hotspot/contact metrics where structures and hotspot annotations are available

## Binder Benchmark

Binder Benchmark evaluates labeled binder/nonbinder datasets through multiple
prediction/refolding engines and post-processing metrics.

Dataset input modes:

* installed Overath 2025 dataset
* repo-format ZIP/path with `input.csv` and `input_pdbs/*.pdb`
* simple ESMFold2 CSV
* metric table only

Installed Overath dataset behavior:

* target filtering
* all-target selection
* per-record checkbox selection
* select-all for currently filtered records
* exact checked-record count before launch
* no hidden first-N truncation when records are selected

Benchmark engines:

* AlphaFast AF3
* ColabFold
* AF2 target-only initial guess
* ESMFold2
* Boltz-2
* RF3
* Protenix
* BoltzGen Fold

Benchmark settings include:

* input mode: `pdb_only`, `hybrid`, or `seq_only_csv`
* generated model input folders for compatible engines
* shared target MSA source and repository/cache paths
* AlphaFast/MMseqs GPU settings
* per-engine recycles, templates, diffusion/sampling steps, samples, and seeds
* ColabFold model count and target-template settings
* target-only initial-guess behavior for AF2: the staged prediction PDB contains
  only the selected target, while the binder is supplied separately as sequence
* target MSA controls for Boltz-2, ESMFold2, RF3, and Protenix

Benchmark metric layers:

* engine-native confidence
* recalculated PAE/interface metrics
* Rosetta/PyRosetta structure metrics
* PyMOL geometry/interface metrics
* merged benchmark metrics
* per-engine and merged feature ranking
* average precision and AUROC when both binder and nonbinder classes exist

Benchmark result features:

* queued local worker execution
* active phase/engine progress metadata
* runtime estimate before launch
* per-engine canonical artifact folders
* merged result tables
* result plots and feature summaries
* target-by-engine benchmark matrix rebuilt from completed source runs or a saved matrix umbrella
* matrix includes installed Overath targets even when a target/engine cell is still missing
* matrix umbrella jobs that persist the chosen source run for each engine-target cell
* exact source-run selection for creating target/engine benchmark collections
* canonical markers so newer runs can replace older or unsuitable engine setups per cell
* benchmark collections that combine completed runs
* feature preset save/load support from result pages

## Capacity Benchmark

Capacity Benchmark is a controlled refolding stress-test workflow for estimating
practical engine size limits on the available GPUs.

Phase-one capacity testing supports:

* a folding/refolding capacity test that selects one real sequence from a
  target-library chain, including prepared/cropped/imported targets and installed
  benchmark targets, then increases the multimer copy count, for example 2-mer,
  3-mer, 4-mer, without inventing new biological sequences
* a folding target-panel mode that lists target chains with amino-acid length,
  source category, benchmark provenance, and target-MSA readiness, then tests
  multiple selected targets in one umbrella run
* a separate design-capacity scaffold that keeps one fixed prepared/cropped
  target and increases the requested/tested binder length, with visible design
  generator selection and workload setup for phase-two execution
* optional synthetic target-length stress mode for controlled total-residue
  scaling when no biological target is desired
* AlphaFold DB PDB import in Target Preparation, so AFDB proteins can be
  prepared, trimmed, and given target-chain MSAs before capacity tests
* capacity target tables show whether the shared sequence-hashed target MSA cache
  is already available for AF3, ColabFold, Boltz-style, RF3, Protenix, and
  ESMFold2 practical/full runs
* target preparation caches shared target MSAs for both the raw selected import
  chains and the cleaned/trimmed prepared target, so imported target-library rows
  and prepared rows can both be reused by downstream engines; missing entries are
  filled through the local AlphaFast/MMseqs GPU Docker path and stored in the
  shared sequence-hashed A3M repository
* one queued refolding-evaluation child job per engine/system-size cell, so one
  OOM or timeout does not stop unrelated cells
* all standard refolding engines exposed by the refolding evaluator:
  AlphaFast AF3, ColabFold, AF2 target-only initial guess, ESMFold2, Boltz-2,
  RF3, Protenix, and BoltzGen Fold
* explicit run-depth presets:
  capacity-only minimal inference/generation, practical full prediction settings
  with cached/prepared target MSAs but without heavy post-processing, and
  full-workflow validation/post-processing
* folding-capacity cells run globally one at a time on the selected GPU, ordered
  from shorter to longer systems
* capacity-only AF3 runs use explicit query-only/empty MSAs, avoiding GPU
  MMseqs searches inside the measured folding stage; practical/full profiles
  require cached/prepared target MSAs where the engine supports them
* non-model failures are classified separately and retried once; only model-load,
  inference, or output OOM failures stop larger cells for that engine
* sampled peak and total GPU memory are stored per capacity cell
* separate or all-engine execution from one page
* parent capacity-result aggregation with per-cell status, failure reason,
  sequence length, copy count, total residues, and a success/failure heatmap
* in-place extension of an existing matrix with omitted engines or larger copy
  counts, plus cell-level recalculation without creating another parent run
* larger copy-count extensions launch only for engines whose nearest smaller
  cell succeeded; engines that already failed receive explicit inherited-failure
  cells without consuming GPU time
* explicit umbrella-run selection in Results, scoped profiles per target/run
  depth, and practical follow-up umbrellas seeded at each engine's largest
  successful capacity-only system
* the Capacity Benchmark page keeps a compact umbrella-run index; each
  umbrella's matrix, plots, capacity profile, extension controls, and practical
  follow-up actions live on its dedicated result page
* capacity heatmap cells are selectable and link directly to the corresponding
  child prediction result
* RF3, Protenix, and BoltzGen Fold benchmark adapters require a successful
  engine subrun with at least one predicted structure; empty/failed subruns
  propagate failure instead of producing a completed result with a blank viewer
* RF3 inputs receive a query-only MSA fallback for every chain without a real
  MSA, preventing an empty recycle stack when MSA searching is disabled
* RF3 capacity-only runs use at least two recycles because Foundry's one-recycle
  path can produce an empty recycle stack even for valid query-only MSA inputs
* ColabFold benchmark runs require at least one predicted PDB/mmCIF even when
  `colabfold_batch` exits with code zero; capacity tests disable target templates
  to avoid template-cardinality errors in repeated-sequence multimers
* normal Jobs-page pause, stop, resume, and delete behavior for the child jobs

## Design Campaigns

Design Campaigns is a separate, queued multi-engine task. It leaves the legacy
Design page unchanged and currently supports:

* one prepared target and a selected target-chain set
* RFdiffusion classic, BindCraft, RFdiffusion3 / Foundry, BoltzGen, PXDesign,
  Genie3, ESMFold2 binder design, Protpardelle-1c, and Proteina-Complexa
* an interactive Mol* target viewer with chain filtering and click-to-edit hotspots
* per-engine vanilla workflow settings
* shared design attempts, sequences-per-backbone, and random-seed controls
* binder-template redesign mode from an existing target-binder PDB complex:
  choose target and binder chains, inspect the complex in Mol*, and use the
  uploaded structure as a normal campaign candidate source
* template sequence redesign through the shared MPNN refinement stage, including
  full-binder redesign, non-interface redesign, redesign of selected residues,
  or locking of selected residues while preserving the input backbone
* an execution-plan preview showing how shared workload settings map to each engine
* sequential execution on one assigned GPU
* child-job provenance under one parent campaign
* continue-after-failure behavior
* campaign history opens through the central Results page, matching Benchmark
  and Refolding result navigation
* pass-first harmonization with native-rank and confidence fallbacks
* a configurable survivor cap per workflow
* common BindCraft-compatible validation across generators:
  two-model AF2 target-template prediction, two-model binder-only AF2 checks,
  binder RMSD, ipSAE, DSSP binder/interface secondary-structure percentages,
  design-reference hotspot RMSD, confidence gating, PyRosetta relaxation,
  interface scoring, and common filtering
* BindCraft no-filter candidate-pool mode during common validation
* layered per-engine candidate pools for common validation:
  native survivors/ranked rows first, then available prefilter/generated
  structures for RFdiffusion3 / Foundry, BoltzGen, PXDesign, Genie3,
  Protpardelle-1c, and Proteina-Complexa
* BindCraft default parity filters for model 1, model 2, and average binder-loop
  content (maximum 90%) and hotspot RMSD (maximum 6 A); missing parity metrics
  fail explicitly
* exact BindCraft trajectory references when available, with native emitted
  complexes recorded as the equivalent design reference for other generators
* an executable shared sequence-refinement stage using ProteinMPNN, LigandMPNN,
  or Soluble ProteinMPNN
* full-binder redesign or non-interface redesign that fixes binder residues
  within 5 Å of the target in each harmonized complex
* configurable sequences per structure, sampling temperature, omitted amino
  acids, and deterministic campaign seed reuse for sequence refinement
* refinement provenance, skipped-candidate reporting, and explicit marking that
  redesigned sequences require post-refinement validation
* a dedicated campaign Results view with workflow funnel tables, candidate
  metrics, metric scatter plots, per-engine pass-rate bars, selected-candidate
  Mol* viewing, provenance, downloads, logs, and artifact previews
* an executable evaluation stage over final candidates, either using existing
  structures or refolding with AlphaFast AF3, ESMFold2, Boltz-2, or RF3
* optional shared interface/ipSAE calculations, PyRosetta input and predicted
  metrics, and PyMOL interface/secondary-structure metrics
* configurable refolding recycles, sampling steps, samples, target-MSA use, and
  PyRosetta process count
* evaluation child-job provenance plus copied merged/interface/Rosetta tables in
  the parent campaign artifacts
* normalized campaign candidates for downstream sequence design, refolding, and analysis

## Result Pages

Result pages provide:

* contract file display
* task-specific summaries
* native or benchmark-aware result sections
* artifact browsing for non-benchmark jobs
* benchmark metric provenance groups
* original/source rank recovery from normalized candidate provenance, including
  generic `import_rank` and BindCraft-native rank fields
* Mol* structure viewers where structures are available
* result ZIP preparation only after an explicit button click

## Current Caveats

The app is ahead of parts of `APP_BUILD_PLAN.md`. In particular:

* Protpardelle-1c is present in the code and UI, while the older plan still
  describes it as a future integration.
* Some direct design jobs still run synchronously from Streamlit buttons, while
  benchmark/refolding evaluations use queued local workers.
* `core/modules.py` is a compact module registry and may not list every newer
  engine exposed by the full benchmark/refolding pages.
* Some tool integrations depend on local images, checkpoints, and reference
  folders under `/mnt/db/reference_files`.
* External source folders such as `tools_to_implement/` may be intentionally
  untracked but required for Docker image builds.
