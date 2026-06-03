# mn-protein-design App Build Plan

`mn-protein-design` is a small Docker-orchestrated Streamlit workbench for local protein binder design. The app should stay closer to `mn-ligand` than to the heavier OVO architecture: task pages, file-backed jobs, strict artifacts, and clear result pages. OVO remains useful as a reference for working pipelines and refolding behavior, but the app should be self-contained and should not call OVO directly.

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

A **task** is a user-facing action such as target preparation, hotspot detection, design, sequence design, refolding, or analysis. A **job** is one frozen run with inputs, parameters, Docker image, status, logs, result files, and provenance. A **tool run** is the raw execution of one backend. A **candidate set** is the normalized handoff between tools.

The current design philosophy is:

```text
Vanilla tool pipeline first
  Preserve each tool's intended workflow, defaults, native metrics, and native result tables.

Normalize outputs second
  Convert native outputs into candidates.jsonl with target/binder chain roles, structures, sequences, and metrics.

Common validation only when useful
  Run app-level monomer refolding, complex refolding, IPSAE/iPAE scoring, and ranking only when it adds information or when the user explicitly sends a native result set through Refolding / Validation.
```

Do not hide native tool behavior behind a single generic wizard. Each design backend gets its own tab on the Design task page, with shared target/hotspot inputs at the top and tool-specific settings inside the tab.

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

The Design page is the home for generator-specific vanilla workflows. Shared controls include:

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

The app should not force all tools into the same pipeline shape. Native end-to-end tools keep their own native analysis. Generator-only tools feed the common downstream modules.

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
| Design / foundation model | ESM repo in `tools_to_implement/esm` | ESM/ESM3 tooling; potential sequence or structure generation/scoring | Not implemented; no smoke test yet | Must inspect repo and define whether it is a design backend, scoring backend, or utility backend before adding UI |
| Sequence design | ProteinMPNN / LigandMPNN / Foundry MPNN | common sequence redesign | Implemented where needed | Keep model choice explicit; default to protein sequence design unless tool requires ligand-aware weights |
| Refolding | AF2 initial guess | target-template complex refolding | Implemented | Default complex validation route |
| Refolding | Boltz2 | monomer and optional complex validation | Implemented | Use model/cache in `/mnt/db/reference_files/boltz_models` |
| Analysis | IPSAE / iPAE / interface metrics | common app re-analysis | Implemented for normalized refolding outputs | Only meaningful when PAE matrices and chain roles are available |

## Vanilla Tool Implementation Pattern

For each new design backend, implement in this order:

1. Inspect the upstream repo documentation and examples.
2. Identify the intended vanilla workflow.
3. Identify required model/cache folders under `/mnt/db/reference_files`.
4. Build or select the Docker image.
5. Create a workflow wrapper in `mn_protein_design/workflows/<tool>.py`.
6. Add a Design page tab with only the useful user-facing parameters.
7. Show the exact payload/files sent to the algorithm.
8. Preserve native outputs under `artifacts/raw/<tool>/`.
9. Normalize outputs into `artifacts/normalized_candidates/candidates.jsonl`.
10. Add native table support on the Analysis page.
11. Add native Mol* viewer support for accepted/passing designs.
12. Decide whether app re-analysis adds information.

If the tool is native end-to-end and already produces ranked/refolded designs, do not automatically run app re-analysis. Instead, show native results. If the user wants IPSAE/iPAE or common AF2/Boltz2 comparisons, they can send the native candidates through Refolding / Validation.

## Testing Pattern

Smoke tests are useful for Docker mechanics, but not enough to define app behavior. For every new tool:

```text
1. Docker smoke
   Does the image run with local model/cache mounts?

2. Vanilla app-style run
   Use a prepared PDL1 target or another known target.
   Run the same settings a user would enter in the Design page.

3. Artifact contract check
   Confirm input.json, metadata.json, command.json, stdout.log,
   stderr.log, result.json, raw artifacts, and candidates.jsonl exist.

4. Native analysis check
   Confirm native result tables and native Mol* viewer work.

5. Optional validation check
   If useful, send native/generated candidates through Refolding / Validation.

6. App analysis check
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

If Protpardelle-1c is generator-only, it should follow the RFdiffusion pattern:

```text
Protpardelle generation
  -> normalize backbones/complexes
  -> sequence design
  -> monomer refolding
  -> AF2/Boltz2 complex refolding
  -> app analysis
```

If it is not a target-conditioned binder design tool, keep it out of the main Design page until its role is clear.

## ESM Repo Next Steps

`tools_to_implement/esm` is newly present and has not been smoke-tested. Treat it as an unknown capability, not automatically as a binder design backend.

Before implementation:

* inspect `tools_to_implement/esm/README.md`, `tools_to_implement/esm/_assets/ESM3_README.md`, and examples
* decide whether the app use case is sequence generation, structure prediction, variant scoring, inverse folding, or embeddings
* identify model weights, license/API requirements, and whether local weights are available
* define the Docker image and local cache contract
* create a minimal smoke test because none exists yet
* only then decide where it belongs:

```text
Design tab
  if it can generate target-conditioned binder candidates.

Sequence Design tab
  if it is best used for sequence generation/redesign.

Refolding / Validation tab
  if it is best used as ESMFold-like structure prediction.

Analysis tab
  if it is best used for embeddings or sequence/structure scoring.
```

Do not expose ESM in the UI until its role and model/cache requirements are clear.

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
```

The Jobs page is the single job index. It should be filterable by task group, job type, tool, and status, with newest jobs first. Deletion must be explicit and should warn about downstream dependent jobs, but deleting one analysis job should not force deletion of its upstream design run.

## Design Principles

* Keep Streamlit pages thin: pages create configs and display results; workflow modules execute jobs.
* Keep Docker commands inside workflow wrappers and manifests, not page files.
* Preserve native outputs and native metrics for every tool.
* Normalize every backend into `candidates.jsonl`.
* Keep native result views separate from app re-analysis views.
* Track chain roles dynamically instead of relying on fixed chain IDs.
* Keep target-template refolding target-only unless a future mode explicitly asks for complex templates.
* Store enough metadata to reconstruct provenance and downstream relationships.
* Prefer local model caches under `/mnt/db/reference_files` and avoid silent downloads.
* Add one tool at a time: vanilla workflow, normalized outputs, native analysis, optional validation.
