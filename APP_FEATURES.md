# mn-protein-design Available Features

This document inventories the features currently exposed by the app and wired
in the codebase. See `APP_feature.md` for product architecture and
`APP_DEVELOPMENT.md` for engineering conventions and handoff notes.

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
* Design Campaigns
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
* background local-worker execution for candidate-set monomer-refolding checks
* on-demand result ZIP generation from result pages
* deletion planning with optional downstream-job detection
* job codes for compact UI references
* runtime estimates for benchmark/refolding engines from completed run history

## Target Preparation

Target Preparation is the central target hub for importing, inspecting, editing,
masking, and target-only refolding. It is organized as tabbed workflows for
import/preparation, target structure analysis, mutation/masking, and target
refolding.

The target workflow supports:

* uploaded local PDB/mmCIF-style structures
* target chain selection
* chain-aware trimming/preparation
* noncanonical residue reporting/handling hooks
* PDB numbering and chain mapping artifacts
* imported, prepared, cropped, trimmed, mutated, and benchmark target categories
* table-based target selection with category filters and search
* prepared target artifacts that downstream design, detection, refolding, and
  campaign pages consume

Target structure analysis supports:

* chain-level target-library inspection across imported, trimmed, cropped,
  mutated, benchmark, and split-fragment targets
* residue-fragment counting and residue-number gap reporting
* coordinate-break detection from peptide-backbone geometry, including C-N and
  CA-distance checks, so continuous numbering is not confused with continuous
  structure
* warning filters for targets with discontinuous coordinates or residue gaps
* Mol* inspection of the selected target, including highlighted break-boundary
  residues in the sequence/structure viewer
* JSON analysis reports for later reuse by target selection, refolding, and
  benchmark pages
* target-chain MSA depth summaries, including full-chain and fragment-level
  entries when fragment-aware target metadata is available
* cached PNG MSA coverage/depth plots using the shared sequence-hashed A3M
  repository, with high-coverage sequences sorted toward the top and per-position
  depth plotted below the coverage image
* creating one derived split-fragment target structure from a broken source
  chain, with fragment chains kept together in a single multi-chain PDB and
  provenance pointing back to the source target
* a target-library cleanup table for removing deletable prepared/imported target
  jobs from the registry; bundled benchmark/reference rows are shown for context
  but are not treated as direct deletion targets

Mutation and masking support includes:

* creating mutated target copies for downstream design/refolding
* lysine masking for crop-exposed or manually selected residues
* crop-parent/reference comparison when a cropped target can be related back to
  a fuller target
* target selectors that expose mutated targets alongside imported, cropped,
  trimmed, and benchmark targets

Target-only refolding supports:

* a tabbed workflow for target selection, engine configuration, launch, and
  results
* all target-folding/refolding engines exposed by the refolding stack
* background job-worker execution so the page can be left while runs continue
* shared MSA repositories and AlphaFast/MMseqs reference data settings
* serial/shared target-MSA preparation to reduce GPU-memory pressure
* explicit warnings when MSA fallback paths are used
* residue-number gap detection for cropped or discontinuous targets
* bulk evidence presets shared with Binder Benchmark: select all, deselect all,
  Template + MSA engines, Template-only engines, MSA-only engines, and
  No MSA + no template
* default fragmented-target handling with separate fragment chains for compatible
  engines, while the app keeps the biological target provenance as one selected
  target structure
* all engines remain manually available for fragmented targets, with
  fragment-chain mode selected by default where applicable
* Protenix-specific split-fragment template staging: target-only fragment chains
  are collapsed into one Protenix input sequence with one coordinate-preserving
  template JSON, because Protenix recenters per-chain templates and would
  otherwise lose the original fragment layout
* MSA sanitization before engine handoff so cached A3M rows are trimmed to
  match-column-only alignments for parsers that require equal row lengths
* target-refolding result tables with multi-row selection and guarded deletion
  of completed/failed/stopped jobs while active jobs remain protected
* benchmark-style result viewing for target-refolding jobs
* template/input CA RMSD, aligned-atom counts, coverage, and structure overlays
  where input/reference coordinates are available

Typical outputs include target PDB, target metadata, residue numbering maps,
viewer state files, target analysis reports, split-fragment target copies,
mutated target copies, and target-refolding artifacts.

## PPI / Hotspot Detection

The PPI / Hotspot Detection page exposes:

* ScanNet PPI/interface scoring
* PeSTo i_v4_1 residue-level protein-interface prediction
* Surf2Spot hotspot prediction
* MaSIF-seed target-surface support
* prepared-structure preview
* existing detection result browsing

Detection jobs write native tool artifacts plus normalized outputs where the
workflow supports them.

PeSTo runs in a CUDA 12.8 / PyTorch 2.7 container suitable for Blackwell-class
GPUs. Its checkpoint is mounted read-only from the shared parameter store at
`/mnt/db/reference_files/pesto/i_v4_1/model_ckpt.pt`; outputs retain the input
chain IDs and residue numbering for downstream hotspot selection.

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

* imported, prepared, cropped, trimmed, mutated, or installed benchmark target
  selection
* target chain selection
* hotspot/epitope selection from text or Mol*
* binder length/range
* design attempts
* campaign name
* per-tool payload preview before launch

Available design tabs:

* RFdiffusion classic
* BindCraft
* BindCraft 2
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
* **BindCraft 2**: full vanilla workflow on the Design page and selectable in
  Design Campaigns → Vanilla multi-engine. Campaign execution uses the native
  BC2 trajectory/sequence design, AF2 validation, ranking, and normalized
  mmCIF candidates. GPU and reserved CPU slots follow the persistent job
  scheduler; AlphaFold weights resolve from the configured reference root.
  The app exposes BC2's 10 binder-format presets, its `induced_fit` and
  `fold_switch` objectives, and all 9 native design-property presets. Modality
  length defaults are preserved unless a BC2-specific override is entered;
  this keeps BC2's peptide, oligomer, large-binder, and multidomain ranges
  independent of the shared length control used by other engines.
  Generator-only integration remains a separate follow-up.
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
* legacy-compatible chain-role import, with old binder/target layouts preserved
  as source metadata and normalized at downstream workflow boundaries
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

Available candidate-set follow-up actions:

* queue binder-only monomer refolding from the selected candidate set
* choose the monomer folding backend (`Boltz-2 monomer` or `ESMFold2 monomer`);
  the older `ESMFold monomer` container is hidden behind an experimental toggle
* choose the GPU device using the same GPU activity panel used by other jobs
* preserve source candidate-set provenance in the queued job metadata, including
  source job/run identifiers, candidate count, and selection rule

The Candidate Sets page only launches monomer-refolding jobs. Ranking, RMSD
filtering, parent-overlay inspection, and candidate triage are handled on the
dedicated result page for the queued monomer-refolding job.

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
* a persistent local worker service for queued jobs; it is started when a UI
  workflow queues work and remains independent of the Streamlit server
* shared CPU-slot admission, exclusive device-specific GPU admission, Docker
  CPU limits, worker health, and visible allocation/wait details

Use `mn-protein-design worker` to run the queue service in a terminal,
`mn-protein-design worker-status` to inspect it, and
`mn-protein-design worker-stop` to stop new dispatch. Already running worker
jobs continue after the service or Streamlit is stopped. Some older workflows
still run synchronously in Streamlit and do not use this service.

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
* OpenFold-3
* Boltz-2 target-template prediction/refolding
* RF3
* Protenix, the legacy PXDesign-backed v0.5 adapter
* Protenix v1, via the standalone Protenix CLI
* Protenix v2, via the standalone Protenix CLI
* BoltzGen Fold

Monomer-only binder refolding:

* can be launched from an existing normalized candidate set
* runs as a normal queued `refolding-validation` job with `job_type` set to
  `monomer_refolding`
* supports the wired monomer backends `Boltz-2 monomer` and `ESMFold2 monomer`;
  legacy `ESMFold monomer` remains available only as an experimental backend
* writes normalized monomer-refolding candidates for downstream inspection
* records parent/source candidate provenance in each normalized candidate where
  available
* calculates binder monomer RMSD against the parent/source binder coordinates
  when a parent complex is available
* records monomer confidence metrics from the selected backend, including
  ESMFold2 pTM/plDDT or Boltz-2 confidence JSON values when produced
* normalizes complex-refolding prediction structures at app workflow boundaries:
  engines may use native chain layouts internally, while normalized candidates
  expose target/reference chains as `A`, `B`, `C`, ... and binder/design chains
  as `Z`, `Y`, `X`, ... where role metadata is available
* records chain-role findings separately from true warnings, so legacy-compatible
  `A`-binder/`B`-target data remains usable without being treated as an error

Engine settings include:

* GPU device
* evidence preset buttons: Template + MSA, Template-only, MSA-only, and
  No MSA + no template
* shared target MSA source
* MSA repository path
* AlphaFast/MMseqs database path
* requirement for real target MSAs
* per-engine model/cache/checkpoint paths
* recycles, diffusion/sampling steps, samples, seeds, and template toggles
* selected target PDB coordinates plus binder sequence as the default AF2 input
* explicit legacy whole-complex AF2 initial-guess controls
* ESMFold2 sequence-only, initial-guess/template-like, target-MSA, and combined
  initial-guess plus target-MSA modes; the model consumes target MSAs when
  enabled and consumes the selected target structure through its
  initial-guess/distogram conditioning path when that mode is selected
* target-template and full-PAE controls for Boltz-2
* OpenFold-3 checkpoint path, target-MSA use, diffusion samples, model seeds,
  recycles, and optional MSA server use
* Protenix v1/v2 model selection, target-MSA use, optional template use,
  Pairformer cycles, diffusion steps, and samples

OpenFold-3 uses the local CUDA 13 / Pixi container
`mn-openfold3:cu13`, intended for RTX 5090 / Blackwell-class hosts.
Its default checkpoint is stored in the shared parameter store at
`/mnt/db/reference_files/openfold3/of3-p2-155k.pt`. For benchmark targets, the
app reuses the shared target MSA repository by sequence match and attaches only
target-chain main MSAs; binder chains remain MSA-free.

Standalone Protenix v1/v2 uses the local CUDA 12.8 / PyTorch container
`mn-protenix:cu128`, intended for RTX 5090 / Blackwell-class hosts.
The Protenix CLI cache, downloaded model files, and related parameters live
under `/mnt/db/reference_files/protenix` and are mounted into the container as
`/ref/protenix` through `PROTENIX_ROOT_DIR`. The current default v1 model is
`protenix_base_default_v1.0.0`, with `protenix_base_20250630_v1.0.0` available
as an alternate; the v2 default is `protenix-v2`. Template-capable target-only
split-fragment refolding is staged as one collapsed Protenix chain with one
combined template so fragment geometry is preserved.

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

Monomer-refolding result pages provide a separate layout focused on binder-only
fold sanity checks:

* ranked monomer table with selectable ranking metric and direction
* RMSD threshold filtering on the result page, defaulting to 3.5 A when RMSD is
  available
* rank scatter plot for the currently selected monomer ranking metric
* Mol* structure viewer overlaying the monomer fold with the parent/source
  design complex when available
* stored monomer RMSD plus viewer alignment RMSD for the selected structure
* parent/source design path and monomer prediction path for traceability

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
* OpenFold-3
* Boltz-2
* RF3
* Protenix, the legacy PXDesign-backed v0.5 adapter
* Protenix v1, via the standalone Protenix CLI
* Protenix v2, via the standalone Protenix CLI
* BoltzGen Fold

Benchmark settings include:

* input mode: `pdb_only`, `hybrid`, or `seq_only_csv`
* bulk evidence presets shared with Target Refolding: select all, deselect all,
  Template + MSA engines, Template-only engines, MSA-only engines, and
  No MSA + no template
* generated model input folders for compatible engines
* chain-role compatibility handling for old benchmark rows and explicit
  boundary normalization for newly staged candidate/refolding inputs
* shared target MSA source and repository/cache paths
* AlphaFast/MMseqs GPU settings
* per-engine recycles, templates, diffusion/sampling steps, samples, and seeds
* ColabFold model count and target-template settings
* target-only initial-guess behavior for AF2: the staged prediction PDB contains
  only the selected target, while the binder is supplied separately as sequence
* target MSA controls for Boltz-2, ESMFold2, RF3, legacy Protenix, and
  standalone Protenix v1/v2
* ESMFold2 is included in the template-capable presets through its
  initial-guess/distogram conditioning mode, and in the MSA-capable presets
  through per-target-chain A3M inputs
* Protenix v1/v2 model, template, Pairformer-cycle, diffusion-step, and sample
  controls, using `/mnt/db/reference_files/protenix` for shared model/cache data

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
* chain-role finding and warning artifacts for staged benchmark inputs, with
  legacy-compatible rows reported as informational findings rather than warnings
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
  is already available for AF3, ColabFold, Boltz-style, RF3, legacy Protenix,
  standalone Protenix v1/v2, and ESMFold2 practical/full runs
* target preparation caches shared target MSAs for both the raw selected import
  chains and the cleaned/trimmed prepared target, so imported target-library rows
  and prepared rows can both be reused by downstream engines; missing entries are
  filled through the local AlphaFast/MMseqs GPU Docker path and stored in the
  shared sequence-hashed A3M repository
* one queued refolding-evaluation child job per engine/system-size cell, so one
  OOM or timeout does not stop unrelated cells
* all standard refolding engines exposed by the refolding evaluator:
  AlphaFast AF3, ColabFold, AF2 target-only initial guess, ESMFold2, Boltz-2,
  RF3, legacy Protenix, standalone Protenix v1/v2, and BoltzGen Fold
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
* RF3, Protenix-family, and BoltzGen Fold benchmark adapters require a
  successful engine subrun with at least one predicted structure; empty/failed
  subruns propagate failure instead of producing a completed result with a
  blank viewer
* RF3 inputs receive a query-only MSA fallback for every chain without a real
  MSA, preventing an empty recycle stack when MSA searching is disabled
* RF3 capacity-only runs use at least two recycles because Foundry's one-recycle
  path can produce an empty recycle stack even for valid query-only MSA inputs
* ColabFold benchmark runs require at least one predicted PDB/mmCIF even when
  `colabfold_batch` exits with code zero; capacity tests disable target templates
  to avoid template-cardinality errors in repeated-sequence multimers
* normal Jobs-page pause, stop, resume, and delete behavior for the child jobs

## Design Campaigns

Design Campaigns is a separate, queued multi-engine task for comparing generator
behavior on the same target and carrying selected outputs through shared
sequence-design, refolding, and metric stages.

Target setup supports:

* imported, prepared, cropped, trimmed, mutated, or benchmark target selection
* table-based target browsing with category filters, search, chain length,
  fragment count, residue-number gap count, source, job code, and PDB path
* a PPI/hotspot-run summary column showing completed detection jobs associated
  with each target
* selecting one or more chains from one target PDB
* interactive Mol* target viewing with chain filtering and click-to-edit hotspots
* detection-guided hotspot suggestions from MaSIF, PeSTo, ScanNet, and Surf2Spot
  when completed detection outputs exist
* threshold controls for detection score, MaSIF score, and minimum side-chain SASA
* per-tool colored detection previews and sequence-strip highlighting
* side-by-side square-ish Mol* panels for detection preview and campaign hotspot
  selection when detection suggestions are active

Shared campaign controls include:

* fixed binder length or binder-length range
* design attempts
* deterministic random seed
* campaign name
* generator support notes for engines that natively accept ranges versus engines
  where the app samples one fixed length per attempt
* an execution-plan preview showing how shared workload settings map to each engine
* sequential execution on one assigned GPU
* child-job provenance under one parent campaign
* continue-after-failure behavior as normal campaign behavior
* campaign history opened through the central Results page

Workflow recipes currently include:

* **Vanilla multi-engine**: run selected generators in their app-supported native
  or vanilla style, then normalize/harmonize outputs into downstream-ready
  campaign candidates.
* **Backbone -> sequence selection**: use selected engines as backbone or
  structure generators, then apply shared sequence design and internal refolding
  screens to compare generator attempts on the same footing.

Available campaign generators include RFdiffusion classic, BindCraft,
RFdiffusion3 / Foundry, BoltzGen, PXDesign, Genie3, ESMFold2 binder design,
Protpardelle-1c, and Proteina-Complexa. In the staged workflow, all generators
are selected by default.

Staged backbone -> sequence selection supports:

* a Design Engines sub-tab containing only generator selection and
  generator-specific parameters
* a Sequence Selection sub-tab containing shared sequence-design and internal
  refolding-screen controls
* level-1 ProteinMPNN-style sequence generation for selected generator attempts
* selectable generator outputs for level-1 sequence design, defaulting to all
  generators except sequence-native ESMFold2 and Proteina-Complexa
* level-1 internal refolding with one selectable refolding engine, defaulting to
  AF2 initial guess
* engine-specific internal refolder parameters and ranking metrics
* target-aligned binder RMSD filtering, defaulting to 3.5 A
* configurable number of passing sequences retained per generator attempt
* optional structure-quality filtering before level-1 and level-2 sequence
  redesign, including minimum secondary-structure element counts; Genie3 can use
  CA-trace geometry elements when PyDSSP sees only coil on reconstructed
  glycine-backbone outputs
* optional level-2 sequence redesign of the best passing level-1 candidate(s)
* level-2 generator-output selection, defaulting to all staged generators
* level-2 internal refolding, ranking, and target-aligned binder RMSD filtering
* optional inclusion of level-1 winners together with level-2 winners in final
  staged outputs
* fragmented/cropped target handling that restricts compatible internal
  refolders to AF2 initial guess and Boltz-2
* staged chain-role normalization for generator outputs: target chains are
  carried as `A`, `B`, `C`, ... and binder/design chains as `Z`, `Y`, `X`, ...
  wherever role metadata is available
* sequence refinement re-reads staged structures after MPNN-style redesign so
  the binder sequence, binder length, target chains, binder chains, and role
  metadata match the actual structure handed to downstream refolding
* default omitted amino acids for staged sequence redesign are `C,X`
* generation-time metrics are calculated for normalized generator outputs,
  including hotspot fraction, binder radius of gyration, target-aligned RMSD
  where available, PyDSSP secondary-structure fractions/elements, and CA-trace
  geometry element counts
* Genie3 CA-only/generated outputs are reconstructed as no-relaxation glycine
  backbones for downstream ProteinMPNN compatibility, viewer display, and
  secondary-structure/geometry metric calculation
* generator-native sequence emitters such as ESMFold2 binder design and
  Proteina-Complexa can bypass level-1 MPNN redesign and enter later staged
  screening/redesign steps with their own generated sequence+structure
  candidates
* engine scout / pilot style campaign runs can execute generator-only outputs
  across all selected engines before committing to a larger downstream campaign
* engine scout result plots include per-workflow plot tabs, manual X/Y axis
  ranges, raw or normalized Rg axes, secondary-structure fraction plots,
  PyDSSP and CA-trace element-count bar plots, and workflow-specific statistics

Campaign evaluation is exposed as **Evaluation Refolding/Metric** and supports:

* optional final refolding of campaign outputs before scoring
* scoring existing workflow outputs directly when final refolding is disabled
* ipSAE/interface scoring, Rosetta/PyRosetta metrics, and PyMOL metrics
* one selectable final refolding engine with engine-specific parameters
* AlphaFast AF3 as the default final refolder for continuous targets
* AF2 initial guess and Boltz-2 restriction for cropped or residue-gap targets
* optional standalone Protenix v1/v2 final refolding for continuous targets,
  using the same `/mnt/db/reference_files/protenix` model/cache store
* evaluation child-job provenance plus copied merged/interface/Rosetta tables in
  parent campaign artifacts
* normalized campaign candidates for downstream sequence design, refolding, and
  analysis

Operational staged-campaign note:

* AF2 initial guess, Boltz-2, RF3, ColabFold, Protenix-family engines,
  OpenFold-3, and ESMFold2 should all be checked through their selected evidence
  mode. A run labeled Template + MSA should have both a target/template path and
  an MSA path where that engine supports them; a run labeled No MSA + no template
  should consume neither.

The campaign Results view provides workflow funnel tables, candidate metrics,
metric scatter plots, per-engine pass-rate bars, selected-candidate Mol* viewing,
provenance, downloads, logs, and artifact previews.

## Result Pages

Result pages provide:

* contract file display
* task-specific summaries
* native or benchmark-aware result sections
* artifact browsing for non-benchmark jobs
* benchmark metric provenance groups
* original/source rank recovery from normalized candidate provenance, including
  generic `import_rank` and BindCraft-native rank fields
* source-vs-normalized chain-role provenance where available, including original
  prediction paths and role-normalized prediction paths
* Mol* structure viewers where structures are available
* result ZIP preparation only after an explicit button click

## Desired / Planned Features

These items are current desired behavior and are not fully implemented unless
listed elsewhere as available features.

### Target Sequence Repair On Import

A future Target Preparation repair step should help when imported PDB/mmCIF
files have missing coordinate segments, residue-number gaps, insertion codes, or
non-natural residues. The desired workflow is:

* read canonical/full sequence from SEQRES or mmCIF entity/polymer fields when
  available
* compare the reported sequence with ATOM/HETATM coordinates and residue
  numbering
* report missing coordinate segments separately from simple residue-number gaps
* map common modified residues, for example MSE to MET, while preserving an audit
  trail of the original residue names
* offer explicit modes such as keep coordinate-only, standardize modified
  residues, insert sequence gaps for downstream awareness, or create a modeled
  repair copy when a modeling backend is available
* store full-sequence, coordinate-sequence, gap, and modification metadata so
  target selection, cropping, refolding, and design pages can warn correctly
* keep destructive or speculative loop rebuilding optional rather than automatic

PDB entries often contain the full polymer sequence in SEQRES/mmCIF metadata, but
that data is not guaranteed to be complete or biologically identical to the
construct. The first useful implementation should therefore be detection,
reporting, standard residue mapping, and provenance capture before any automatic
structure rebuilding.

### Target Table Naming

Target tables currently expose residue-number discontinuities as breaks in some
places. The desired UI wording is to rename this to residue gaps or numbering
gaps so trimmed structures with inherited missing residue numbers are not
confused with newly introduced crop breaks.

## Current Caveats

Some earlier planning notes may not reflect the current implementation. In
particular:

* Protpardelle-1c is present in the code and UI, despite older notes describing
  it as a future integration.
* Some direct design jobs still run synchronously from Streamlit buttons, while
  benchmark/refolding evaluations use queued local workers.
* `core/modules.py` is a compact module registry and may not list every newer
  engine exposed by the full benchmark/refolding pages.
* Some tool integrations depend on local images, checkpoints, and reference
  folders under `/mnt/db/reference_files`.
* Standalone Protenix v1/v2 requires the `mn-protenix:cu128` image
  and keeps model/cache/parameter files under `/mnt/db/reference_files/protenix`.
* External source folders such as `tools_to_implement/` may be intentionally
  untracked but required for Docker image builds.
