# mn-protein-design App Behavior and Build Plan

`mn-protein-design` is a Docker-orchestrated Streamlit workbench for local protein binder design and binder-prediction benchmarking. This document is both:

* the behavioral specification for features already implemented in the app
* the build plan and integration contract for future tools

The app stays closer to `mn-ligand` than to the heavier OVO architecture: task pages, file-backed jobs, strict artifacts, queued GPU work, and clear result pages. OVO remains useful as a reference for working pipelines and refolding behavior, but the app is self-contained and does not call OVO directly.

The core object model is:

```text
Project
  Target
    Task group
      Job
        Tool run(s)
          Artifacts
          Metrics
          Logs
```

A **task** is a user-facing action such as target preparation, hotspot detection, design, sequence design, refolding, analysis, or binder benchmarking. A **job** is one frozen run with inputs, parameters, Docker image, status, logs, result files, and provenance. A **tool run** is the raw execution of one backend. A **candidate set** is the normalized handoff between tools.

The current design philosophy is:

```text
Vanilla tool pipeline first
  Preserve each tool's canonical repository/publication workflow, defaults, native metrics, and native result tables.
  Keep it permanently runnable as the engine's Reference / Vanilla mode, whether it is one end-to-end
  container or a repository-defined chain of several tools.

Normalize outputs second
  Convert native outputs into candidates.jsonl with target/binder chain roles, structures, sequences, and metrics.

Common validation only when useful
  Run app-level monomer refolding, complex refolding, IPSAE/iPAE scoring, and ranking only when it adds information or when the user explicitly sends a native result set through Refolding / Validation.

Composable workflows are additive
  Later, allow explicit combinations of generators, sequence designers, refolders, and evaluators.
  Never silently replace or mutate the engine's Reference / Vanilla mode.
```

Do not hide native tool behavior behind a single generic wizard. Each design backend gets its own tab on the Design task page, with shared target/hotspot inputs at the top and tool-specific settings inside the tab.

The product purpose and feedback loop are:

```text
Prepare target and identify hotspots
  -> generate binder candidates with one or more design engines
  -> redesign sequences where required
  -> refold and evaluate candidates
  -> select candidates for experimental testing
  -> use labeled experimental binder/nonbinder datasets in Binder Benchmark
  -> identify which refolding engines and evaluation features best discriminate success
  -> apply those validated evaluation choices to future design campaigns
```

Binder Benchmark supports binder design; it is not the final product by itself. Its main value is to determine which prediction engines, refolding settings, and evaluation features are useful for prioritizing designs before experimental testing. Benchmark performance must not be presented as proof that an individual design binds.

## Runtime Contract

Every job writes to the file-backed run store:

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

GPU jobs enter the shared GPU queue. A user can continue browsing results and configuring additional jobs while another job runs. Newly submitted GPU jobs wait until the active GPU job finishes.

Result ZIP archives are generated only when the user requests a download. Opening a result page must not synchronously build a large ZIP or block Streamlit interaction.

`metadata.json` must carry enough provenance for downstream filtering and deletion:

```json
{
  "job_code": "A1B2C",
  "task_group": "design",
  "job_type": "design_campaign",
  "tool": "genie3",
  "status": "completed",
  "campaign_name": "PDL1 Genie3 Boltz2 vanilla 1 design",
  "upstream_task_group": "target-prep",
  "upstream_run_id": "20260519-101544-47e30b79",
  "upstream_job_code": "8D2B8"
}
```

`result.json` should use a stable shape:

```json
{
  "success": true,
  "job_type": "design_campaign",
  "tool": "boltzgen",
  "inputs": {},
  "outputs": {
    "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl"
  },
  "metrics": {},
  "downstream_artifacts": {}
}
```

Multi-engine benchmark runs use one visible parent job. Internal engine executions may temporarily use child job mechanics, but they are hidden immediately and consolidated into the parent before completion:

```text
runs/benchmark/<run_id>/
  input.json
  metadata.json
  command.json
  stdout.log
  stderr.log
  result.json
  artifacts/
    engines/
      alphafast_af3/
      af2_initial_guess/
      boltz2/
      colabfold/
      esmfold2/
    benchmark/
      merged_benchmark_metrics.csv
      merged_benchmark_feature_ranking.csv
      predicted_rosetta_metrics.csv
      pymol_files/
```

All final paths in CSV, JSON, JSONL, PML, YAML, and result metadata must point to this canonical parent layout. Historical `command.json` files retain the exact paths used during execution.

## Task Pages

### Target Preparation

Purpose:

* upload or download PDB/mmCIF structures
* select target chains
* trim each selected chain independently
* detect non-standard residues and optionally repair them
* optionally run PDBFixer for non-canonical residue replacement and missing atom cleanup
* preserve true PDB numbering, insertion codes, chain IDs, and viewer index mappings

Outputs:

```text
target_trimmed.pdb
target.json
residue_numbering_map.json
noncanonical_residues.json
viewer_state.json
```

### PPI / Hotspot Detection

Purpose:

* ScanNet residue-level PPI/interface prediction
* Surf2Spot hotspot prediction
* MaSIF-seed surface/patch workflow when the local image contract is stable

Outputs:

```text
residue_scores.csv
scored_target.pdb
hotspot_clusters.json
viewer_state.json
```

### Target Cropping

Cropping is separate from target preparation. Preparation trims selected chains; cropping happens after PPI/hotspot detection and can combine:

* manual residue ranges
* Mol* residue picks from sequence or structure
* sphere expansion around selected residues
* plane/slice selection when implemented

Outputs:

```text
target_cropped.pdb
crop_selection.json
residue_numbering_map.json
viewer_state.json
```

### Design

The Design page is the home for binder-design engines and their canonical reference workflows. Every engine must provide a permanent **Reference / Vanilla** mode that reproduces the repository- or publication-defined end-to-end workflow. That workflow may be:

* a self-contained end-to-end container, as with BindCraft
* a repository-defined chain of tools, such as RFdiffusion followed by its intended sequence-design and validation stages
* a native sequence-design, prediction, filtering, and ranking workflow

The app may later provide a **Composable / Custom** mode for explicitly mixing generators, sequence designers, refolders, and evaluators. Custom mode supplements the reference workflow; it never becomes a hidden replacement for it.

Shared controls include:

* prepared target or cropped target
* hotspot selection from Mol*
* binder length or range
* design attempt count
* campaign name

Each tool tab owns its native parameters and payload preview. The user should be able to inspect what files and settings are sent to the algorithm.

Current tabs:

* RFdiffusion classic
* RFdiffusion3 / Foundry
* BindCraft
* BoltzGen
* PXDesign
* Genie3
* Protpardelle-1c
* Proteina-Complexa

The app should not force all tools into the same pipeline shape. Reference / Vanilla mode preserves the complete upstream-defined workflow, including any multi-tool chain. Composable / Custom mode uses normalized candidates to connect compatible stages while clearly recording every substitution.

### Sequence Design

Purpose:

* run ProteinMPNN, LigandMPNN, Foundry MPNN, or tool-specific sequence redesign
* rename sequences per backbone as `<backbone_id>_mpnn_001`, `<backbone_id>_mpnn_002`, etc.
* produce normalized sequence-designed candidates

Outputs:

```text
redesigned_candidates.jsonl
redesigned_sequences.fasta
sequence_design_scores.csv
```

### Refolding / Validation

Purpose:

* always run monomer refolding before complex refolding when the user starts validation from this page
* use Boltz2 monomer refolding by default where available
* use AF2 initial guess as the default complex refolding mode
* make Boltz2 target-template complex refolding selectable
* use target-template behavior: the target template is the target protein only, not the accepted binder pose
* write normalized prediction artifacts and full PAE matrices when available so app-level IPSAE/iPAE scoring is possible

Outputs:

```text
monomer_refolding/
complex_refolding/
validated_candidates.jsonl
refolding_metrics.csv
```

Key metrics:

* monomer RMSD against the designed backbone
* target-aligned binder pose RMSD for complex refolding
* pLDDT, pTM/ipTM, PAE/iPAE
* Boltz2 confidence/PDE-style metrics where available
* hotspot contact recovery
* IPSAE-derived scores when PAE matrices are available

### Analysis

The Analysis page must distinguish two result types:

```text
Native Pipeline Results
  Tool-specific tables and viewers from the vanilla workflow.
  Examples: BindCraft accepted designs, BoltzGen all_designs_metrics.csv,
  PXDesign native summary, Proteina-Complexa native binder CSV, Genie3 native results.

App Re-analysis Results
  Common app ranking after Refolding / Validation has produced normalized
  complex predictions and PAE matrices.
```

Native end-to-end tools should show their native table and native Mol* viewer, then stop. The app re-analysis section should not appear for native runs unless the user has explicitly selected a refolding/validation candidate set. This avoids mixing native tool scores with app-level refolding scores.

App re-analysis should show:

* live filters above the table
* ranked candidate table
* structure viewer with target and designed/predicted binder overlays
* plots for useful numeric metrics

### Binder Benchmark

Purpose:

* evaluate labeled binder/nonbinder datasets with several prediction/refolding engines
* compare engine-native confidence values with standardized interface scores
* calculate shared Rosetta and PyMOL structure metrics
* rank features by average precision and AUROC when both classes are present
* record runtime per engine so future benchmark duration can be estimated
* determine which validation engines, settings, and metrics should be used to prioritize candidates from binder-design campaigns

Benchmark results should inform configurable validation presets, for example:

```text
PDL1 validation preset
  AF2 initial guess + ESMFold2
  target-only initial guess
  rank primarily by selected interface metric
  retain Rosetta/PyMOL metrics as secondary filters
```

Presets derived from benchmarks must record their source benchmark run, target scope, dataset composition, feature direction, AP/AUROC, and chosen threshold. A preset learned on one target or dataset must not silently be described as universally valid.

The benchmark page has five tabs:

```text
Dataset
  Select installed Overath 2025 records, upload repo-format data,
  provide a simple ESMFold2 CSV, or inspect a precomputed metric table.

Engines
  Select prediction engines and configure their native parameters,
  model/reference folders, shared compute device, and MSA behavior.

Metrics
  Select standardized interface metrics, Rosetta evaluation, and PyMOL evaluation.

Run
  Review exact record count, engines, settings, and estimated runtime before submission.

Results
  Browse benchmark jobs and open their unified result pages.
```

#### Dataset Selection Behavior

The installed Overath dataset is not selected by taking the first N rows.

* The user filters by one or more targets, or selects all targets.
* The filtered record table contains a checkbox in front of every row.
* A select-all checkbox selects or clears every currently filtered record.
* The generated input CSV contains exactly the checked rows.
* The Run tab reports the exact checked-record count.
* No hidden engine-level row limit truncates the checked selection.
* Missing/`nan` target IDs are not exposed as selectable target names.

#### Engines

Currently integrated benchmark engines:

* AlphaFast AF3
* ColabFold / AF2-Multimer
* AF2 initial guess
* Boltz-2 target-template refolding
* ESMFold2 initial-guess prediction

Default binder-benchmark behavior:

* target chains may use an existing cached MSA or AlphaFast MMseqs2-GPU generation
* binder chains use no MSA
* initial-guess/template modes condition the target only, never the binder or accepted interface
* multi-chain target roles come from each dataset record's `binder_chain`/`binder_chains` and `target_chains`
* prediction and PAE-based confidence preserve separate target chains
* Rosetta and PyMOL evaluation PDBs normalize the binder group to chain `A` and the complete target group to chain `B`, so their `A_B` interface means binder versus all declared target chains
* engine-native defaults remain visible and configurable
* model/reference directories live in the corresponding engine settings
* common compute settings and MSA settings are separate from engine-specific settings

#### Benchmark Metric Layers

The result page separates metric provenance instead of presenting implementation CSVs as peers:

```text
Engine-Native Prediction Confidence
  Values emitted directly by each engine or its native parser.
  Examples: ranking score, pTM, ipTM, pLDDT, ipLDDT, ipDE, actifpTM.

Standardized Interface Scores
  Shared post-processing applied to every compatible prediction.
  Examples: ipSAE, LIS, pDockQ, pDockQ2, interface PAE.
  Higher is better for ipSAE/LIS/pDockQ/pDockQ2; lower is better for interface PAE.

Structure-Based Evaluation
  Rosetta interface energy/packing metrics and PyMOL geometry metrics,
  displayed separately per engine.

Complete Analysis Export
  One wide merged table joining metadata and every metric by binder ID.
```

The workflow may retain separate adapter source CSVs for traceability, but the default UI combines equivalent standardized interface metrics into one cross-engine table.

#### Ranking and Plots

Each engine produces:

```text
<engine>_metrics.csv
<engine>_feature_benchmark.csv
<engine>_feature_summary.json
```

The merged result also produces:

```text
merged_benchmark_metrics.csv
merged_benchmark_feature_ranking.csv
merged_benchmark_feature_summary.json
```

Feature ranking uses labeled binder/nonbinder records. Average precision and AUROC require at least one binder and one nonbinder. For one-class runs, ranking CSVs remain valid header-only files and the UI explains why ranking is unavailable.

Plots include:

* top features by average precision
* AP versus AUROC
* binder/nonbinder separation ranked by the selected feature
* precision/recall by rank
* ROC curve

Repeated aliases and setup/status features are hidden by default but remain available for technical inspection.

#### Benchmark Runtime and Progress

The visible parent benchmark job reports the active phase and engine:

```text
input preparation
ESMFold2
AF2 initial guess
Boltz-2
ColabFold
AlphaFast AF3
standardized interface metrics
Rosetta / PyMOL evaluation
merge and ranking
```

`result.json` records runtime by engine, candidate count, and total residue count. The Run tab uses completed benchmark history to estimate future runtime.

## Normalized Candidate Contract

Every backend should emit `artifacts/normalized_candidates/candidates.jsonl`.

Candidate shape:

```json
{
  "candidate_id": "rfdiffusion_classic_00001_protein_mpnn_001",
  "source_tool": "rfdiffusion_classic",
  "result_kind": "normalized",
  "stage": "complex_refolding",
  "parent_candidate_id": "rfdiffusion_classic_00001",
  "target_chains": ["A"],
  "binder_chains": ["B"],
  "target_pdb": "artifacts/target.pdb",
  "design_model_pdb": "artifacts/raw/rfdiffusion/output/design_00001.pdb",
  "complex_pdb": "artifacts/raw/af2_initial_guess/output/design_00001_af2ig.pdb",
  "binder_sequence": "MSEQUENCE...",
  "hotspots": ["A54", "A56", "A58"],
  "metrics": {
    "binder_plddt": 88.2,
    "iptm": 0.76,
    "ipae": 5.8,
    "ipsae": 0.42,
    "monomer_rmsd": 1.2,
    "binder_rmsd": 2.4
  },
  "raw_metadata": {}
}
```

Important chain rule:

* never assume chain `A` is always binder or target
* infer target chains from the selected prepared/cropped target
* infer binder chains as the non-target chains in generated complexes
* carry `target_chains` and `binder_chains` through every downstream job

## Current Tool Status

| Group | Tool | Role | Current status | Implementation philosophy |
| --- | --- | --- | --- | --- |
| Target prep | internal Python / BioPython / PDBFixer | trim, repair, non-canonical residue handling | Implemented scaffold and active workflow | Keep target chain/numbering mapping canonical |
| PPI detection | ScanNet | PPI/interface scoring | Implemented | Native Docker result plus normalized artifacts |
| Hotspot detection | Surf2Spot | hotspot prediction | Implemented | Requires cleaned structures; repair non-canonical residues before use |
| Surface matching | MaSIF-seed | surface patch/seed search | Partly wired; image is `masif_seed:latest` | Keep as detection/hotspot support, not design |
| Design | RFdiffusion classic | backbone generator | Implemented with full vanilla-style downstream app pipeline | Generator -> MPNN -> monomer -> AF2 initial guess -> analysis |
| Design | RFdiffusion3 / Foundry | backbone generator | Implemented with full downstream app pipeline | Same normalized behavior as RFdiffusion classic |
| Design | BindCraft / FreeBindCraft | native end-to-end binder workflow | Implemented | Use native accepted designs and native CSVs; app validation is separate |
| Design | BoltzGen | native end-to-end design workflow | Implemented | Use `all_designs_metrics.csv` from `final_ranked_designs` for native table |
| Design | PXDesign | native end-to-end design workflow | Implemented | Show native summary; app re-analysis only after explicit validation |
| Design | Proteina-Complexa | native end-to-end design workflow | Implemented | Use native CSV; rank by native iPAE-like value when present, otherwise ipTM |
| Design | Genie3 | native end-to-end workflow with ColabFold or Boltz2 option | Implemented | Use `/mnt/db/reference_files/genie3`, `/mnt/db/reference_files/alphafold_models`, and `/mnt/db/reference_files/boltz_models` caches |
| Design | Protpardelle-1c | motif-conditioned or protein-design generator | Not implemented in app yet | Next candidate if its vanilla binder workflow is clear |
| Foundation model | ESM repo in `tools_to_implement/esm` | broader ESM/ESM3 sequence, generation, and scoring capabilities | Partly implemented through ESMFold2 refolding/benchmarking; broader capabilities remain unassigned | Add further capabilities only after defining whether they belong to design, sequence design, validation, or analysis |
| Sequence design | ProteinMPNN / LigandMPNN / Foundry MPNN | common sequence redesign | Implemented where needed | Keep model choice explicit; default to protein sequence design unless tool requires ligand-aware weights |
| Refolding | AF2 initial guess | target-template complex refolding | Implemented | Default complex validation route |
| Refolding | Boltz2 | monomer and optional complex validation | Implemented | Use model/cache in `/mnt/db/reference_files/boltz_models` |
| Benchmark | AlphaFast AF3 | complex prediction and native AF3 confidence | Implemented | Target MSA through cache/MMseqs2-GPU; binder has no MSA |
| Benchmark | ColabFold | AF2-Multimer benchmark prediction | Implemented | Uses `/mnt/db/reference_files/alphafold_models` and shared target MSA |
| Benchmark | AF2 initial guess | target-conditioned complex refolding | Implemented | Target-only initial guess; binder/interface template disabled |
| Benchmark | Boltz-2 | target-template complex refolding | Implemented | Target-only template; full PAE available |
| Benchmark | ESMFold2 | complex prediction with target initial guess | Implemented | Native confidence plus standardized interface adapter |
| Benchmark evaluation | Rosetta / PyRosetta | interface energy and packing | Implemented | Calculated per engine and displayed per engine |
| Benchmark evaluation | PyMOL | interface geometry and secondary structure | Implemented | Calculated per engine and for input complexes |
| Analysis | IPSAE / iPAE / interface metrics | common app re-analysis | Implemented for normalized refolding outputs | Only meaningful when PAE matrices and chain roles are available |

## Binder Design Engine Implementation Contract

Binder design is the app's primary workflow. Adding a design engine requires preserving its intended method while making its candidates usable by the shared sequence-design, validation, analysis, and benchmarking-informed prioritization layers.

### 1. Classify the Design Engine

Before implementation, classify the engine as one of:

```text
Backbone generator
  Produces binder backbones or target/binder complexes but requires sequence design.
  Typical downstream path: generation -> ProteinMPNN -> validation -> analysis.

Sequence design engine
  Redesigns sequences for existing backbones or complexes.
  Typical downstream path: sequence design -> validation -> analysis.

Native end-to-end binder design
  Generates, sequences, predicts/refolds, filters, and ranks designs internally.
  Typical downstream path: native results -> optional explicit app validation.

Motif/scaffold generator
  Produces scaffolds around a motif but may not perform target-conditioned binder design.
  Keep outside the main Design page until a binder-design workflow is demonstrated.
```

This classification describes the engine's core capability, not necessarily the default workflow exposed to the user. A backbone generator can still have a repository-defined reference binder-design pipeline that chains it with sequence design, prediction, filtering, and ranking. In the UI, a design engine may therefore represent a complete reference pipeline rather than only one executable.

Document whether the engine actually performs target-conditioned binder design and document the complete upstream reference workflow around it. Do not expose a general protein generator as a binder-design engine merely because it can produce proteins.

### 2. Reproduce the Vanilla Workflow First

Inspect the upstream publication, repository documentation, examples, and default configurations. Identify:

* required target representation and chain assumptions
* hotspot, epitope, motif, or contig input format
* supported binder lengths and design counts
* whether sequence design is internal or external
* whether structure prediction/refolding is internal or external
* canonical stage order, including any external tools invoked by the upstream workflow
* native filtering and ranking stages
* native accepted/passing/failing result definitions
* required model weights, databases, licenses, and reference paths

Reference / Vanilla mode is a permanent product mode, not merely an implementation milestone. It should reproduce the intended upstream workflow and scientific defaults, with only necessary local path, container, and compute adaptations. If the upstream workflow invokes several tools, preserve that chain intact.

Composable / Custom mode may be added after Reference / Vanilla mode works. It must never silently alter reference defaults, stage choices, filters, requested counts, or result semantics.

### 3. Docker and Reference Contract

Build or select an image that:

* runs the upstream workflow without modifying its scientific defaults unnecessarily
* supports the intended local GPUs, including RTX 4090 and RTX 5090 where applicable
* mounts model/reference folders from `/mnt/db/reference_files`
* avoids silent duplicate model downloads
* writes outputs into the mounted job directory
* records the exact deterministic command in `command.json`
* returns non-zero on failure

Keep engine-specific model/reference paths in the engine's Design tab settings. Shared compute controls belong in shared controls only when they truly apply across engines.

### 4. Design Page Contract

Each design engine receives its own tab on the Design page.

Each tab defaults to:

```text
Reference / Vanilla
  Runs the canonical repository/publication workflow and defaults.
  Shows the complete stage chain, even when several tools are orchestrated.
  Serves as the reproducible baseline for comparison, troubleshooting, and future development.

Composable / Custom
  Optional explicit workflow composition using compatible generators, sequence designers,
  refolders, evaluators, and filters.
  Clearly labeled as non-reference and records every component, version, setting, and substitution.
```

Shared inputs:

* prepared or cropped target
* selected target chains
* hotspot/epitope residues from Mol*
* binder length or range when supported
* number of design attempts
* campaign name

Engine tab requirements:

* expose useful native parameters with sensible upstream defaults
* make Reference / Vanilla the default mode and keep it available after custom composition is implemented
* show the complete reference stage chain and identify which stages run inside a container or through app orchestration
* explain unsupported shared inputs by disabling them rather than silently ignoring them
* preview the exact payload, files, chain mapping, and command-relevant settings
* clearly distinguish design attempts from requested accepted designs
* estimate runtime when historical timing data exists
* visibly label Composable / Custom jobs and warn when they deviate from the reference workflow

Do not force every engine into identical controls or pipeline stages.

### 5. Workflow Runner and Job Contract

Implement execution in `mn_protein_design/workflows/<tool>.py` or the established shared design workflow. Streamlit page code creates configuration and renders results; it does not assemble Docker commands.

The job must record:

```text
input.json
  target, hotspots, chain roles, user parameters, model/reference paths

metadata.json
  tool, job type, campaign name, status, queue/progress, upstream provenance

command.json
  exact executed commands

stdout.log / stderr.log
  complete execution logs

result.json
  native outputs, normalized outputs, metrics, downstream artifacts
```

Long GPU design jobs enter the shared GPU queue and report meaningful phases such as generation, sequence design, prediction, filtering, and final ranking.

### 6. Preserve Native Outputs and Semantics

Store untouched or minimally staged native outputs under:

```text
artifacts/raw/<tool>/
```

Preserve:

* native accepted/passing/failing classifications
* native ranking and filter tables
* native predicted structures and sequences
* native confidence and energy metrics
* native configuration and intermediate outputs needed for provenance

Never rename a native score into a shared score merely because the names look similar. Keep metric direction and scale documented.

### 7. Normalize Design Candidates

Every design engine must produce:

```text
artifacts/normalized_candidates/candidates.jsonl
artifacts/normalized_candidates/campaign_result.json
```

Each candidate must carry:

* stable `candidate_id`
* `source_tool` and stage
* parent candidate relationship when sequence design or filtering creates descendants
* target and binder chain roles
* target, design-model, binder, and complex structure paths when available
* binder sequence when available
* requested hotspots/epitope
* native metrics under unambiguous names
* raw metadata needed to trace the native result

Candidate IDs must remain stable across downstream sequence design, refolding, analysis, and export.

### 8. Choose the Correct Workflow Mode

Reference / Vanilla mode always runs the complete canonical workflow:

```text
BindCraft reference
  -> run the complete native BindCraft pipeline in its container
  -> preserve native filtering, ranking, and accepted designs

RFdiffusion reference
  -> run the repository-defined RFdiffusion binder-design chain
  -> preserve its intended sequence-design, prediction, filtering, and ranking stages

Other reference engines
  -> run their complete repository/publication workflow
  -> preserve native outputs and conclusions
```

Composable / Custom mode explicitly connects compatible stages through normalized candidates:

```text
selected generator
  -> normalize candidates
  -> selected sequence designer
  -> selected monomer and/or complex refolder
  -> selected evaluators and filters
  -> custom ranking
```

For example, a custom workflow may use RFdiffusion generation, LigandMPNN sequence design, Boltz-2 refolding, and selected Rosetta/PyMOL evaluation. The app must record component lineage, versions, settings, and parent candidate relationships. It must label the result as custom and keep it distinct from the RFdiffusion reference result.

Both modes normalize their outputs so results can be compared or passed into later tasks. Do not automatically overwrite native/reference conclusions with a different app validation pipeline.

### 9. Design Results and Analysis UI

The native result page must provide:

* native summary and status counts
* native ranked/filter table
* structure viewer for selected candidates
* paths/downloads for native outputs
* clear selection action for sending candidates to Sequence Design or Refolding / Validation

App re-analysis appears only after candidates pass through the shared validation workflow. It must remain visually distinct from native analysis.

When benchmark evidence has identified useful validation engines or features, the app may offer those as named validation presets. The user must still see which benchmark run and metrics informed the preset.

### 10. Binder Design Engine Integration Test

Before calling a design engine implemented:

```text
1. Reproduce a small upstream Reference / Vanilla example with the canonical commands, stages, and defaults.
2. Confirm a repository-defined multi-tool reference chain remains intact.
3. Confirm Reference / Vanilla remains available and unchanged after Composable / Custom mode is added.
4. Run an app-style target-conditioned test on PDL1 or another known target.
5. Confirm target chains and hotspot/epitope inputs reach the engine correctly.
6. Confirm requested design attempts and actual produced designs are reported separately.
7. Confirm native output tables, structures, sequences, status classes, and conclusions are preserved.
8. Confirm normalized candidates contain stable IDs and correct chain roles.
9. Confirm native result table and structure viewer work.
10. Run one explicit custom composition when supported and confirm every substituted stage is visibly labeled.
11. Confirm reference and custom runs have distinct provenance and neither mutates the other's defaults.
12. Confirm downstream selection into Sequence Design and Refolding / Validation works.
13. Confirm queued execution, progress phases, logs, and failure states work.
14. Confirm model/reference paths use mounted caches without duplicate downloads.
15. Confirm results remain usable after app restart.
16. Confirm any benchmark-informed preset records its provenance and does not claim experimental success.
```

### 11. Current Design-Engine Code Touchpoints

The normal design-engine integration touches:

```text
mn_protein_design/app/pages/design.py
  Design tab, shared inputs, native settings, payload preview, and submission.

mn_protein_design/workflows/<tool>.py
  Native runner, Docker command, output parser, normalization, progress, and result contract.

mn_protein_design/workflows/design.py
  Shared design orchestration and downstream handoff where applicable.

mn_protein_design/core/candidates.py
  Shared candidate normalization only when the established contract needs extension.

mn_protein_design/app/pages/analysis.py
mn_protein_design/app/pages/results.py
  Native tables/viewers and downstream app-analysis rendering.

containers/<tool>/
  Dockerfile and runtime documentation when no suitable image already exists.
```

Prefer extending established workflow and candidate helpers over adding tool-specific behavior to shared pages.

## Benchmark Engine Implementation Contract

Adding a benchmark engine is more than placing another checkbox on the page. A complete integration must cover execution, parsing, normalization, evaluation, artifacts, UI, runtime tracking, and tests. Its purpose is to improve evidence-based prioritization of candidates produced by the design workflows.

### 1. Define the Engine Role

Document:

* whether the engine predicts a complex, refolds an existing complex, or only scores a structure
* whether it accepts PDB, sequence, MSA, template, or initial-guess inputs
* whether target-only conditioning is supported
* native confidence outputs and their directionality
* available structure files and PAE-like matrices
* model weights, databases, licenses, and expected reference paths

Do not label a post-processing metric calculator as a prediction engine.

### 2. Docker and Reference Contract

The image must:

* run on the supported local GPUs, including RTX 4090 and RTX 5090 where applicable
* use mounted reference/model folders instead of silently downloading duplicate weights
* write all outputs into the mounted job directory
* expose a deterministic command suitable for `command.json`
* return a non-zero exit code on failure

Engine-specific model and reference paths belong in the engine settings. Shared GPU and MSA controls belong in shared Compute and MSA sections.

### 3. Workflow Runner

Implement the runner in `mn_protein_design/workflows/`, not in the Streamlit page.

The runner must:

* accept a staged benchmark dataset or normalized candidates
* preserve binder IDs as the merge key
* preserve binder/target chain roles
* apply target-only initial guess/template conditioning unless explicitly configured otherwise
* register as an internal benchmark sub-run when it reuses an existing workflow
* report progress and runtime timing
* record image, parameters, paths, and command provenance

Internal child mechanics must not create extra visible benchmark jobs. Before the parent finishes, engine artifacts are adopted into:

```text
artifacts/engines/<engine>/
```

### 4. Native Output Parser

Create a parser that emits one row per binder ID:

```text
artifacts/benchmark/<engine>_metrics.csv
```

Requirements:

* use a stable engine prefix for native metrics
* include `binder_id`, `candidate_id` when available, and `label`
* preserve native scores without silently changing their scale
* expose paths to predicted structures and confidence arrays when useful
* distinguish missing metrics from numeric zero
* write normalized candidates when the engine produces structures

The parser must also create:

```text
<engine>_feature_benchmark.csv
<engine>_feature_summary.json
```

Use the common ranking helper so one-class datasets produce valid empty ranking tables and a clear explanation.

### 5. Shared Interface Adapter

If the engine provides a predicted complex and PAE-like confidence:

* adapt its structure and confidence outputs to the shared interface scorer
* calculate ipSAE, LIS, pDockQ, pDockQ2, and interface PAE where mathematically valid
* prefix all columns with the engine identifier
* add the values to the combined cross-engine interface display

Do not invent unavailable PAE values from pLDDT or geometry alone.

### 6. Rosetta and PyMOL Evaluation

If a predicted complex can be converted to a usable PDB:

* stage it in `artifacts/benchmark/predicted_metric_pdbs/<engine>/`
* include it in the shared Rosetta calculation
* include it in the shared PyMOL calculation
* display Rosetta and PyMOL results in per-engine tables

Rosetta and PyMOL are evaluation layers, not benchmark engines.

### 7. Merge and Canonical Artifact Paths

Add the engine metric tables to the parent merge using `binder_id`. Verify:

* no duplicate binder rows
* no duplicate columns
* no accidental `_x`/`_y` columns
* no all-null engine block when the engine succeeded
* missing records remain missing rather than shifting row alignment

After engine artifacts move into the canonical parent layout, rewrite path-valued references exactly once. The rewrite must be idempotent and must not modify historical `command.json` files.

### 8. Benchmark UI

Add:

* engine selection control
* a compact engine settings section with sensible defaults
* model/reference paths in that engine section
* native metric explanation and table
* engine ranking-table entry
* standardized interface metrics when supported
* Rosetta/PyMOL per-engine display when supported
* runtime estimate and progress label

The UI must explain metric provenance. Native confidence, standardized interface scores, and structure-based evaluation must remain visually separate.

### 9. Engine Integration Test

Before calling an engine implemented:

```text
1. Run one labeled binder and one labeled nonbinder.
2. Confirm the engine produces one native metric row per input record.
3. Confirm predicted structures are viewable and non-empty.
4. Confirm native confidence values are parsed.
5. Confirm shared interface metrics when supported.
6. Confirm Rosetta and PyMOL metrics when supported.
7. Confirm merged metrics contain the engine block without duplicate rows/columns.
8. Confirm engine ranking is selectable and AP/AUROC are calculated.
9. Confirm all non-historical artifact references exist.
10. Confirm only one visible parent benchmark job remains.
11. Confirm runtime and progress information are recorded.
12. Run a one-class test and confirm the UI explains that ranking is unavailable.
```

### 10. Current Code Touchpoints

The normal benchmark-engine integration touches:

```text
mn_protein_design/app/pages/benchmark.py
  Engine selection and settings, exact dataset selection, run summary,
  runtime estimate, and workflow submission.

mn_protein_design/workflows/benchmark.py
  Input staging, runner invocation, native parser, feature ranking,
  shared metric adapters, artifact consolidation, merge, and progress.

mn_protein_design/workflows/refolding.py
  Reusable complex-refolding runner when the engine is also available
  from Refolding / Validation.

mn_protein_design/app/pages/results.py
  Native metric table, ranking selector, standardized interface display,
  per-engine Rosetta/PyMOL display, plots, and output paths.

mn_protein_design/core/jobs.py
  Only when new internal-job, queue, or provenance behavior is required.

containers/<engine>/
  Dockerfile and engine-specific runtime documentation when no suitable
  image already exists.
```

Prefer extending the existing helpers and contracts over adding engine-specific behavior directly to page code.

## Testing Pattern

Smoke tests are useful for Docker mechanics, but not enough to define app behavior. For every new tool:

```text
1. Docker smoke
   Does the image run with local model/cache mounts?

2. Vanilla app-style run
   Use a prepared PDL1 target or another known target.
   Run the complete canonical repository/publication workflow through Reference / Vanilla mode.
   Confirm the mode remains the default reproducible baseline.

3. Artifact contract check
   Confirm input.json, metadata.json, command.json, stdout.log,
   stderr.log, result.json, raw artifacts, and candidates.jsonl exist.

4. Native analysis check
   Confirm native result tables and native Mol* viewer work.

5. Optional validation check
   If useful, send native/generated candidates through Refolding / Validation.

6. Optional composable-mode check
   When supported, substitute one compatible stage and confirm the job is clearly
   labeled custom without changing Reference / Vanilla defaults.

7. App analysis check
   Confirm app re-analysis appears only for normalized validation outputs,
   not for native end-to-end design pages unless explicitly selected.
```

Test outputs can live under `smoke_tests/` while debugging. Do not commit smoke outputs, `tools_to_implement/`, `ui_inspiration/`, or workdir runs.

## Protpardelle-1c Next Steps

`tools_to_implement/protpardelle-1c` is not implemented in the app yet. Before adding the tab:

* inspect the README and examples for the intended vanilla run
* determine whether it generates binder backbones, sequences, complexes, or only motif-conditioned scaffolds
* determine whether it supports target-conditioned binder design directly
* identify required model files and Docker image
* run a small PDL1-style app test if target-conditioned binder design is supported
* normalize any produced structures into the shared candidate contract

If Protpardelle-1c is generator-only, first preserve any repository-recommended end-to-end binder-design chain as its Reference / Vanilla mode. Only then expose an optional composable path such as:

```text
Protpardelle generation
  -> normalize backbones/complexes
  -> sequence design
  -> monomer refolding
  -> AF2/Boltz2 complex refolding
  -> app analysis
```

If it is not a target-conditioned binder design tool, keep it out of the main Design page until its role is clear.

## ESM / ESMFold2 Current Role and Next Steps

The local ESM parameter set and ESMFold2 workflow are implemented for complex refolding/validation and binder benchmarking.

Current behavior:

* ESMFold2 predicts target/binder complexes
* benchmark mode supports target initial-guess conditioning
* native confidence, pTM/ipTM, pLDDT, and PAE-derived values are parsed
* predictions enter standardized interface scoring
* predictions enter Rosetta and PyMOL evaluation
* ESMFold2 produces its own selectable ranking table like the other engines

The broader ESM repository may still support additional sequence generation, variant scoring, inverse folding, embeddings, or future binder-design workflows. Those capabilities remain separate future integrations. Do not describe ESMFold2 benchmarking as direct binder generation.

## Package Layout

```text
mn_protein_design/
  run_app.py
  cli.py
  app/
    pages/
      jobs.py
      target_preparation.py
      ppi_detection.py
      target_cropping.py
      design.py
      sequence_design.py
      refolding.py
      analysis.py
      benchmark.py
      results.py
    components/
      molstar_viewer.py
  core/
    jobs.py
    manifests.py
    docker_runner.py
    artifacts.py
    structures.py
    residue_selection.py
    candidates.py
    job_graph.py
  workflows/
    target_prep.py
    scannet.py
    surf2spot.py
    design.py
    rfdiffusion.py
    rfdiffusion3.py
    bindcraft.py
    boltzgen.py
    pxdesign.py
    proteina_complexa.py
    genie3.py
    protpardelle.py
    esm.py
    proteinmpnn.py
    af2_initial_guess.py
    boltz2_refolding.py
    analysis.py
    benchmark.py
```

## Navigation Model

Keep the app task-based:

```text
Jobs

Tasks
  Target Preparation
  PPI / Hotspot Detection
  Target Cropping
  Design
  Sequence Design
  Refolding / Validation
  Analysis
  Binder Benchmark
```

The Jobs page is the single job index. It should be filterable by task group, job type, tool, and status, with newest jobs first. Deletion must be explicit and should warn about downstream dependent jobs, but deleting one analysis job should not force deletion of its upstream design run.

## Design Principles

* Keep Streamlit pages thin: pages create configs and display results; workflow modules execute jobs.
* Keep Docker commands inside workflow wrappers and manifests, not page files.
* Preserve native outputs and native metrics for every tool.
* Normalize every backend into `candidates.jsonl`.
* Keep native result views separate from app re-analysis views.
* Keep engine-native benchmark confidence separate from standardized interface and structure-based evaluation.
* Track chain roles dynamically instead of relying on fixed chain IDs.
* Keep target-template refolding target-only unless a future mode explicitly asks for complex templates.
* Store enough metadata to reconstruct provenance and downstream relationships.
* Prefer local model caches under `/mnt/db/reference_files` and avoid silent downloads.
* Add one tool at a time: permanent Reference / Vanilla workflow, normalized outputs, native analysis, then optional composable workflows and validation.
* Treat core capability classification separately from default app behavior; a generator may still have a multi-tool canonical reference pipeline.
* Never silently turn a custom combination into an engine's reference workflow.
* Keep multi-engine benchmarks as one visible parent job with canonical per-engine artifact folders.
* Select benchmark records explicitly; never silently benchmark the first N rows.
* Use benchmark evidence to improve design-candidate prioritization, while keeping benchmark performance distinct from experimental binding evidence.
* Record the benchmark provenance of any recommended validation preset or ranking feature.
