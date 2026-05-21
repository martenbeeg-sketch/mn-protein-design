# mn-protein-design App Build Plan

`mn-protein-design` should be a small Docker-orchestrated Streamlit workbench, closer in spirit to `mn-ligand` than to OVO. OVO is useful for understanding typed pipelines and downstream artifact compatibility, but v1 should avoid its heavier database, plugin, and Nextflow architecture.

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

A **task** is the user-facing action: prepare a target, detect a PPI interface, predict hotspots, define a design region, design binders, refold candidates, analyze results, or triage candidates. A **job** is one submitted run with frozen inputs, parameters, Docker image, status, logs, outputs, and provenance. A **pipeline** is a job composed of multiple tool runs with typed handoff artifacts.

For binder design, the main v1 abstraction should be a **design campaign**: a parent design job that can run one or more backend tools, normalize their outputs, and optionally apply common downstream sequence design, refolding, filtering, and ranking.

## Task Groups

### Target Preparation

Purpose:

* upload or download PDB/mmCIF files
* select chains
* remove ligands, waters, alternate locations, or unwanted residues
* trim/crop around chains, residue ranges, interfaces, or hotspot regions
* normalize chain IDs, residue numbering, insertion codes, and residue selections
* preserve a mapping between viewer sequence indices and true PDB residue numbering

Outputs:

```text
target_clean.pdb
target_trimmed.pdb
target.json
residue_numbering_map.json
viewer_state.json
```

### PPI / Binding Site Detection

Purpose:

* ScanNet residue-level PPI/interface probabilities
* Surf2Spot hotspot or nanobody epitope predictions
* MaSIF target surface generation or patch search when the container contract is stable

Outputs:

```text
residue_scores.csv
scored_target.pdb
hotspot_clusters.json
viewer_state.json
```

### Hotspot / Design Region Definition

Purpose:

* manual residue picking in Mol* or py3Dmol
* import hotspot residues from ScanNet, Surf2Spot, or MaSIF
* cluster and rank candidate patches
* create design-region hints for downstream tools
* define avoid residues, preferred contact residues, and optional fixed interface residues
* export tool-neutral hotspot and contig-hint files

Outputs:

```text
hotspots.json
selected_patch.json
contig_hint.json
interface_constraints.json
viewer_state.json
```

### Backbone / Binder Design

Purpose:

* generate binder backbones or complexes against a prepared target and design region
* support both single-tool design runs and multi-tool design campaigns
* support native tool pipelines as well as synchronized benchmarking pipelines
* normalize raw tool outputs into a shared candidate artifact format
* support lightweight design-only runs first, then fuller sequence/refold pipelines

Initial tools:

* RFdiffusion classic
* RFdiffusion3 / Foundry
* BindCraft / FreeBindCraft
* BoltzGen
* Genie3
* Proteina-Complexa
* PXDesign
* Protpardelle-1c

Outputs:

```text
designed_complexes/
binder_only/
sequences.fasta
raw_model_metadata.json
normalized_candidates/
candidates.jsonl
campaign_result.json
```

### Design Campaigns

Purpose:

A **design campaign** is a parent design job that can run several binder-design tools against the same target, hotspot set, binder-length constraints, and common downstream validation settings. It allows the app to answer two different questions:

```text
How well does each tool perform as originally intended?
How well do different generators perform under the same downstream evaluation pipeline?
```

Supported campaign modes:

```text
Native mode
  Run each tool close to its original pipeline and preserve native outputs/metrics.

Synchronized benchmark mode
  Run tools to comparable checkpoints, normalize outputs, then apply common
  ProteinMPNN, refolding, filtering, and ranking.

Hybrid mode
  Keep native tool outputs, but also send normalized candidates through the common
  downstream validation and optional redesign pipeline.
```

Common design parameters:

```text
target_pdb
target_chains
hotspots
avoid_residues
binder_length_min
binder_length_max
num_backbones_or_candidates
num_sequences_per_backbone
random_seed
use_soluble_mpnn
run_monomer_validation
run_complex_validation
run_minimization_or_relaxation
run_interface_fixed_redesign
common_filter_preset
```

Tool-specific parameters remain available in per-tool tabs, but all selected tools should emit normalized candidates whenever possible.

Campaign outputs:

```text
campaign_config.json
tool_runs/
raw_candidates/
normalized_candidates/
sequence_design/
refolding/
analysis/
ranked_designs.csv
campaign_result.json
```

### Sequence Design / Refinement

Purpose:

* ProteinMPNN or LigandMPNN when not embedded in the design backend
* optional soluble ProteinMPNN redesign for all candidates, including tools that already produced a sequence
* interface-aware redesign where contacting residues can be fixed and the rest redesigned
* side-chain reconstruction, relaxation, clash cleanup, and optional scoring
* keep PyRosetta-dependent analysis optional so the app can run with lighter open tooling first

Outputs:

```text
redesigned_sequences.fasta
redesigned_candidates.jsonl
relaxed_structures/
sequence_design_scores.csv
fixed_positions.json
```

### Refolding / Structure Prediction Validation

Purpose:

* independently repredict designed sequences and complexes after sequence design
* run AF2 initial-guess refolding, following the OVO `AlphaFoldInitialGuess` pattern
* run Boltz2 initial-guess refolding, following the newer OVO Boltz refolding workflow
* compare the refolded structure to the design model, target, motif, hotspot residues, and binder pose
* produce confidence metrics that downstream analysis can rank and filter

Inputs:

```text
target_clean.pdb
designed_complex.pdb
redesigned_sequences.fasta
hotspots.json
candidates.jsonl
```

Outputs:

```text
af2_refolded/
boltz2_refolded/
refolding_metrics.csv
refolding_result.json
validated_candidates.jsonl
```

Key metrics:

* AF2 pLDDT, pTM/ipTM, PAE, interface PAE
* AF2 target-aligned binder RMSD and binder/monomer RMSD
* Boltz2 pLDDT, PDE/ipDE-style confidence metrics where available
* Boltz2 design RMSD, binder RMSD, motif RMSD, target-aligned binder RMSD
* hotspot contact recovery and interface contact preservation
* optional iPAE/iPSAE-style interface-confidence metrics when available

### Analysis / Filtering

Purpose:

* interface metrics
* pLDDT, PAE, ipTM-like, PDE, and ipDE-like metrics where available
* iPAE/iPSAE-style interface scoring where available
* shape complementarity, buried SASA, contacts, hotspot contacts
* clash and minimization checks
* monomer-vs-design agreement
* diversity clustering
* final ranked design table
* compare pass rates across tools and campaign modes

Outputs:

```text
metrics.csv
ranked_designs.csv
tool_summary.csv
selected_design_bundle/
analysis_result.json
```

## Tool Matrix

| Group                  | Tool                                            | Role                                                 | v1 Status                              |
| ---------------------- | ----------------------------------------------- | ---------------------------------------------------- | -------------------------------------- |
| Target preparation     | internal Python / BioPython / PDBFixer          | clean, crop, trim, validate, residue-number mapping  | Implement first                        |
| PPI detection          | ScanNet                                         | residue-level PPI / epitope / IDP binding prediction | Container smoke passed                 |
| Hotspot detection      | Surf2Spot                                       | PPI hotspot and nanobody epitope clusters            | Container smoke passed                 |
| Surface matching       | MaSIF-seed                                      | surface patch / seed search                          | Later, after contract stabilization    |
| Binder design          | RFdiffusion classic                             | established binder/scaffold generation               | Smoke passed using OVO image           |
| Binder design          | RFdiffusion3 / Foundry                          | modern RFdiffusion backend                           | Smoke passed using Foundry image       |
| Binder design          | BindCraft / FreeBindCraft                       | fuller binder workflow baseline                      | BindCraft smoke passed using OVO image |
| Binder design          | BoltzGen                                        | design-only and inverse-folding/filtering workflow   | PDL1 design-only smoke passed          |
| Binder design          | PXDesign                                        | target parsing and backbone design                   | Container smoke passed                 |
| Binder design          | Protpardelle-1c                                 | motif-conditioned binder backbone design             | Container smoke passed                 |
| Binder design          | Proteina-Complexa                               | advanced binder generation backend                   | PDL1 smoke recreated                   |
| Binder design          | Genie3                                          | generative backbone backend                          | Container smoke passed                 |
| Sequence design        | ProteinMPNN / soluble ProteinMPNN               | common sequence-design and redesign backend          | Add after first design adapter         |
| Refolding / validation | AF2 initial guess                               | independent refolding and confidence metrics         | Borrow contract from OVO               |
| Refolding / validation | Boltz2 initial guess                            | modern structure prediction/refolding validation     | Borrow contract from OVO               |
| Analysis               | IPSAE, Rosetta-style metrics, interface metrics | ranking and filtering                                | Add after design contracts             |

The strongest implementation rule is to make artifact contracts boring and strict before adding too many tools. Protein design tools all have their own worldview, so the app should normalize them into a small shared vocabulary: target, chains, residues, hotspot set, design region, candidate structure, sequence, metrics, rank.

## Normalized Candidate Contract

Every binder-design backend should be normalized into a shared candidate representation. Some tools produce only backbones, some produce sequences, some produce complexes, and some produce native metrics. The app should not force them to behave identically at generation time; instead, it should normalize their outputs before common downstream stages.

Each campaign should write:

```text
normalized_candidates/
  candidates.jsonl
  candidate_000001/
    design_model.pdb
    complex.pdb
    binder_only.pdb
    sequence.fasta
    metadata.json
```

Candidate JSON shape:

```json
{
  "candidate_id": "rfdiffusion_classic_000001",
  "campaign_id": "design_2026_05_20_001",
  "source_tool": "rfdiffusion_classic",
  "source_mode": "synchronized",
  "stage": "normalized",
  "parent_candidate_id": null,

  "target_pdb": "artifacts/target.pdb",
  "design_model_pdb": "normalized_candidates/candidate_000001/design_model.pdb",
  "complex_pdb": "normalized_candidates/candidate_000001/complex.pdb",
  "binder_pdb": "normalized_candidates/candidate_000001/binder_only.pdb",
  "sequence_fasta": "normalized_candidates/candidate_000001/sequence.fasta",

  "target_chains": ["A"],
  "binder_chain": "B",
  "hotspots": ["A:150", "A:154", "A:188"],
  "binder_length": 85,

  "has_backbone": true,
  "has_sequence": false,
  "has_complex": true,

  "metrics": {},
  "metadata": {}
}
```

Common downstream stages should consume `candidates.jsonl` instead of tool-specific folders.

## Tool Capabilities

Each binder-design manifest should declare capabilities so that the campaign engine can decide which downstream stages are needed.

Example capability fields:

```yaml
capabilities:
  produces_backbone: true
  produces_sequence: false
  produces_complex: true
  supports_hotspots: true
  supports_binder_length: true
  supports_fixed_interface: false
  supports_native_pipeline: true
  supports_synchronized_pipeline: true
  supports_hybrid_pipeline: true
  has_native_metrics: false
```

The campaign engine can then apply simple rules:

```text
If candidate has no sequence -> run ProteinMPNN.
If candidate has sequence and hybrid mode is enabled -> optionally create solubleMPNN variants.
If candidate has no independently predicted complex -> run AF2/Boltz2 validation.
If common analysis is enabled -> score every candidate with the same filters.
```

## Proposed Package Layout

```text
mn_protein_design/
  run_app.py
  cli.py
  app/
    pages/
      jobs.py
      jobs_target.py
      jobs_detection.py
      jobs_design.py
      target_preparation.py
      ppi_detection.py
      hotspot_detection.py
      design.py
      refolding.py
      analysis.py
      results.py
    components/
      structure_viewer/
      campaign_config/
  core/
    jobs.py
    manifests.py
    docker_runner.py
    artifacts.py
    structures.py
    residue_selection.py
    candidates.py
    campaign.py
  workflows/
    target_prep.py
    scannet.py
    surf2spot.py
    rfdiffusion.py
    rfdiffusion3.py
    bindcraft.py
    boltzgen.py
    pxdesign.py
    protpardelle.py
    proteina_complexa.py
    genie3.py
    design_campaign.py
    normalize_candidates.py
    proteinmpnn.py
    soluble_mpnn.py
    interface_redesign.py
    af2_initial_guess.py
    boltz2_refolding.py
    candidate_filters.py
  manifests/
    scannet.yaml
    surf2spot.yaml
    rfdiffusion.yaml
    rfdiffusion3.yaml
    bindcraft.yaml
    boltzgen.yaml
    pxdesign.yaml
    protpardelle_1c.yaml
    proteina_complexa.yaml
    genie3.yaml
    proteinmpnn.yaml
    soluble_mpnn.yaml
    af2_initial_guess.yaml
    boltz2_refolding.yaml
mn-protein-design-workdir/
  workdir/
    runs/
```

Tool Dockerfiles stay in the existing top-level `containers/` directory. The Streamlit app should use manifests and workflow wrappers rather than embedding Docker shell snippets directly in page files.

## Manifest Contract

Each tool should have a manifest that captures the app-facing contract:

```yaml
tool: scannet
group: detection
image: mnprot-scannet:latest
gpu: false
inputs:
  target_pdb:
    type: pdb
params:
  mode:
    type: enum
    default: interface
    choices: [interface]
  assembly:
    type: bool
    default: false
outputs:
  scored_pdb:
    path: predictions/*.pdb
    type: pdb
  residue_scores:
    path: predictions/*.csv
    type: residue_score_table
```

Binder-design manifests should extend the same pattern with capabilities and normalized candidate outputs:

```yaml
tool: rfdiffusion_classic
group: design
image: mnprot-rfdiffusion:latest
gpu: true
capabilities:
  produces_backbone: true
  produces_sequence: false
  produces_complex: true
  supports_hotspots: true
  supports_binder_length: true
  supports_native_pipeline: true
  supports_synchronized_pipeline: true
  supports_hybrid_pipeline: true
inputs:
  target_pdb:
    type: pdb
  hotspots_json:
    type: hotspots
  contig_hint:
    type: contig_hint
common_params:
  binder_length_min:
    type: int
    default: 60
  binder_length_max:
    type: int
    default: 120
  num_designs:
    type: int
    default: 100
params:
  contig:
    type: string
    default: ""
  partial_T:
    type: int
    default: 50
  hotspot_res:
    type: residue_list
    default: []
outputs:
  raw_designs:
    path: outputs/*.pdb
    type: raw_candidate_set
  metadata:
    path: outputs/*.json
    type: raw_model_metadata
```

The UI can render common controls from these manifests. Custom pages add the protein-specific UX: chain picking, residue picking, hotspot preview, design-region selection, campaign mode selection, candidate triage, and result ranking.

## Navigation Model

Do not make v1 one giant wizard. Protein design needs iteration and backtracking. Use task pages plus job/result pages:

```text
Jobs
  Target Prep
  Detection
  Design
  Refolding
  Analysis

Tasks
  Prepare Target
  Detect Interface / Hotspots
  Define Design Region
  Run Design Campaign
  Refold / Validate Designs
  Analyze Designs
```

This mirrors the useful `mn-ligand` split between task pages and job/result pages while keeping the workflow lighter than OVO.

## Design Page UX

The design page should create a campaign config rather than directly constructing Docker commands.

Suggested layout:

```text
Run Binder Design Campaign

1. Target
   - select prepared target
   - show target viewer
   - select hotspot/design-region job
   - preview hotspots and selected patch

2. Common design parameters
   - binder length min/max
   - number of candidates per tool
   - number of sequences per backbone
   - random seed
   - common downstream stages

3. Pipeline mode
   - native
   - synchronized benchmark
   - hybrid

4. Tool selection
   - RFdiffusion classic
   - RFdiffusion3 / Foundry
   - BindCraft / FreeBindCraft
   - BoltzGen
   - PXDesign
   - Proteina-Complexa
   - Protpardelle-1c
   - Genie3

5. Tool-specific tabs
   - render only selected tools
   - expose native parameters without mixing them into the common schema

6. Common filters
   - min pLDDT
   - min ipTM or equivalent
   - max interface PAE / iPAE / iPSAE where available
   - max clashes
   - hotspot contact recovery
   - diversity clustering

7. Submit campaign
```

The page should write `campaign_config.json` and call `run_design_campaign(config)`. It should not know individual Docker commands.

## Pipeline Philosophy

Borrow OVO's good part: typed parameters, known outputs, and downstream compatibility.

```text
Target Prep job
  -> produces clean target structure

Hotspot Detection job
  -> consumes clean target
  -> produces hotspot residues

Design Region job
  -> consumes target + hotspot predictions or manual residues
  -> produces hotspots.json, selected_patch.json, contig_hint.json

Design Campaign job
  -> consumes target + hotspot residues + common/tool-specific parameters
  -> produces raw and normalized candidate structures

Sequence Design job or campaign stage
  -> consumes normalized candidates
  -> produces redesigned sequences and side-chain-complete structures

Refolding job or campaign stage
  -> consumes redesigned sequences + target/design structures
  -> produces AF2 or Boltz2 predictions and refolding metrics

Analysis job or campaign stage
  -> consumes design, sequence, and refolding outputs
  -> produces ranked table and selected design bundle
```

Do not adopt Nextflow first. A local Docker runner with metadata files is enough for v1 and matches the `mn-ligand` style.

## Campaign Execution Philosophy

A design campaign should be a parent job with subruns. Each selected backend tool gets a tool-run folder. Common stages get stage folders.

```text
runs/design/<campaign_run_id>/
  input.json
  metadata.json
  command.json
  stdout.log
  stderr.log
  result.json
  campaign_config.json

  tool_runs/
    rfdiffusion_classic/
      input.json
      command.json
      stdout.log
      stderr.log
      result.json
      artifacts/

    bindcraft/
      input.json
      command.json
      stdout.log
      stderr.log
      result.json
      artifacts/

  stages/
    normalize_candidates/
    proteinmpnn/
    soluble_mpnn/
    monomer_validation/
    af2_initial_guess/
    boltz2_refolding/
    minimization/
    interface_redesign/
    analysis/

  artifacts/
    target.pdb
    hotspots.json
    contig_hint.json
    raw_candidates/
    normalized_candidates/
    sequence_design/
    refolding/
    analysis/
```

This keeps campaigns reproducible and keeps native tool logs/artifacts separate from normalized app artifacts.

## Run Folder Contract

Every job should write:

```text
runs/<task-group>/<run_id>/
  input.json
  metadata.json
  command.json
  stdout.log
  stderr.log
  result.json
  artifacts/
    target.pdb
    hotspots.json
    designs/
    refolded/
    metrics.csv
```

Every `result.json` should use a normalized shape:

```json
{
  "success": true,
  "job_type": "design",
  "tool": "rfdiffusion",
  "inputs": {},
  "outputs": {},
  "metrics": {},
  "downstream_artifacts": {}
}
```

Campaign `result.json` should add campaign-level summaries:

```json
{
  "success": true,
  "job_type": "design_campaign",
  "campaign_mode": "hybrid",
  "selected_tools": ["rfdiffusion_classic", "bindcraft", "boltzgen"],
  "inputs": {},
  "outputs": {
    "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
    "ranked_designs": "artifacts/analysis/ranked_designs.csv"
  },
  "metrics": {
    "raw_candidates": 300,
    "normalized_candidates": 270,
    "passed_final_filters": 12
  },
  "tool_summaries": {},
  "downstream_artifacts": {}
}
```

This gives durable handoff without requiring a heavy database. A later DB can index the same files instead of replacing them.

## Implementation Order

1. Scaffold the app package, CLI, Streamlit navigation, runtime workdir, and generic job table.
2. Implement target preparation first; this becomes the canonical artifact source.
3. Make residue numbering and viewer-selection mapping robust, including PDB residue offsets and insertion-code awareness.
4. Add ScanNet as the first detection Docker tool because its output contract is simple: scored PDB plus residue CSV.
5. Add Surf2Spot second because it introduces clustered hotspot outputs.
6. Add manual hotspot selection and patch-selection UI.
7. Define `DesignCampaignConfig`, `Candidate`, and the normalized candidate contract before adding many design tools.
8. Add RFdiffusion classic as the first design backend.
9. Add candidate normalization and a simple candidate results table.
10. Add ProteinMPNN / soluble ProteinMPNN as a common downstream stage.
11. Add a refolding/validation job type with AF2 initial guess and Boltz2 initial guess contracts borrowed from OVO.
12. Add analysis/ranking page that can consume design-only results or refolding-enriched results.
13. Add BindCraft/FreeBindCraft or BoltzGen as fuller end-to-end design backends.
14. Add PXDesign, Protpardelle-1c, Proteina-Complexa, Genie3, and Foundry/RFD3 behind the same manifest/job contract.
15. Add hybrid campaign mode: preserve native outputs while creating common redesigned/refolded variants.
16. Add advanced analysis metrics such as IPSAE and optional Rosetta-style scoring once candidate artifact contracts are stable.

## v1 Boundaries

Keep v1 local and file-backed:

* Docker as the execution boundary
* `/mnt/db/reference_files` as the shared model/reference cache
* `mn-protein-design-workdir/workdir/runs` as the durable job store
* JSON manifests and result files as the compatibility layer
* normalized candidate artifacts as the cross-tool handoff layer
* no Nextflow, no heavy database, no remote scheduler

This gives us a clean spine now and leaves room for heavier orchestration later without redesigning the user-facing workflow.

## Design Principles

* Keep Streamlit pages thin: pages create configs and display results; workflow modules execute jobs.
* Keep Docker commands inside manifests and workflow wrappers, not page files.
* Normalize every design backend into candidate artifacts before common downstream stages.
* Preserve native outputs so tool-specific behavior is not lost.
* Support synchronized benchmarking only after a candidate contract exists.
* Store enough metadata to distinguish native performance from generator-plus-common-pipeline performance.
* Prefer boring files and strict JSON/CSV/PDB contracts over early database complexity.
* Build one stable adapter first, then scale to many tools.
