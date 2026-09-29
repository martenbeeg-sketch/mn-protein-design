# mn-protein-design Product and Feature Architecture

Updated: 2026-09-29

This document describes the current conceptual product shape and the design
ideas worth retaining. `APP_FEATURES.md` remains the detailed feature inventory.

## Product purpose

`mn-protein-design` is a local Streamlit workbench for:

- target preparation, inspection, cropping, masking, and target refolding;
- PPI/interface and hotspot detection;
- single-engine and multi-engine binder generation;
- sequence design and optimization;
- normalized candidate-set import and reuse;
- monomer and complex refolding/validation;
- structural and interface analysis;
- binder/nonbinder benchmarking;
- GPU/refolding capacity benchmarking.

The product should preserve tool-native evidence while converting reusable
outputs into a shared candidate contract. Prediction confidence, structural
metrics, interface metrics, and benchmark discrimination should remain separate
evidence families.

## Current application shape

The visible navigation is:

```text
Jobs
  Jobs
  Settings

Tasks
  Target Preparation
  PPI / Hotspot Detection
  Target Cropping
  Design
  Design Campaigns
  Sequence Design
  Candidate Sets
  Refolding / Validation
  Analysis
  Binder Benchmark
  Capacity Benchmark

Hidden
  Result Details
  Design Capacity Results
```

The app is scientifically broad and already supports many engines. Its main
architecture is a file-backed job store plus normalized candidate JSONL files.
Heavy design, refolding, sequence-design, analysis, detection, and benchmark
jobs run through the separate local queue service. Jobs reserve CPU slots and
their selected GPU; child steps share their parent allocation. Docker receives
thread limits for the reserved CPU slots and applies a quota when supported by
the host cgroup configuration.

## Central scientific contracts

### Normalized candidates

Every generator or refolder should publish normalized candidates containing:

- stable candidate identity and stage;
- source tool and parent candidate lineage;
- target, complex, and binder structure references;
- binder sequence and length;
- explicit binder and target chains;
- hotspots and contig information;
- method-specific metrics;
- raw/native provenance.

This contract allows design, sequence optimization, refolding, analysis, and
benchmarking to compose without teaching every consumer each engine's private
folder layout.

### Explicit chain roles

Target and binder identity must be explicit and preserved across staging,
prediction, scoring, and visualization. The current `A...` target and `Z...`
binder convention is useful as an app boundary, but engines may receive narrow
local compatibility copies when required.

This design is especially important for fragmented targets, multichain targets,
AF2 initial guess, Protenix template staging, PAE-indexed metrics, and
binder-versus-full-target interface scoring.

### Native evidence plus normalized evidence

Native structures, tables, logs, confidence outputs, and engine settings should
remain available. Normalized candidates and merged metrics enable comparison,
but should not erase the original evidence or imply that metrics from different
methods are physically interchangeable.

## Main workflows

### Target preparation

Target Preparation imports structures, selects chains, analyzes residue gaps and
coordinate breaks, supports derived split-fragment and mutated/masked targets,
prepares target MSAs, and exposes target-only refolding.

Future structure repair should begin with sequence/coordinate comparison,
modified-residue reporting, and explicit provenance before speculative automatic
loop rebuilding.

### Hotspot detection and cropping

ScanNet, PeSTo, Surf2Spot, and MaSIF-derived signals can inform hotspot selection.
The design-campaign UI can overlay these suggestions while keeping manual
selection authoritative. Cropping produces derived targets rather than mutating
the source.

### Design and sequence optimization

The app supports native or app-orchestrated paths for RFdiffusion, BindCraft,
RFdiffusion3/Foundry, BoltzGen, PXDesign, Genie3, ESMFold2 binder design,
Protpardelle-1c, and Proteina-Complexa.

Sequence Design consumes normalized backbones and uses LigandMPNN,
ProteinMPNN/SolubleMPNN variants, or Foundry-native MPNN paths. Optional
refolding and analysis remain distinct downstream evidence.

### Design campaigns

Design Campaigns compare generators on the same target and can compose:

```text
target + hotspots
  -> generator children
  -> normalized candidates
  -> sequence redesign
  -> internal refolding filters
  -> optional second redesign
  -> optional final refolding and metrics
```

Campaigns should retain child provenance and continue independent engines after
one engine fails.

### Refolding and analysis

Refolding / Validation consumes normalized candidates and supports multiple
prediction engines with explicit MSA/template evidence modes. Analysis separates
native pipeline results from app-level re-analysis.

### Binder and capacity benchmarks

Binder Benchmark evaluates labeled binder/nonbinder datasets across engines and
metric layers. Capacity Benchmark evaluates practical engine limits using real
sequences and increasing system sizes. Completed, running, queued, failed, and
missing engine-target cells should remain distinct.

## Concepts worth reusing in mn-ligand

These concepts transfer well when adapted to ligand-domain artifacts:

1. **A normalized exchange entity.** Protein candidates demonstrate the value of
   one reusable schema. mn-ligand already follows the stronger typed-artifact
   version of this idea for targets, compound sets, pockets, poses, trajectories,
   and energies.
2. **Explicit biological roles at adapter boundaries.** Chain-role maps are
   analogous to preserving receptor, ligand, parent compound, formulation, and
   pose identities in mn-ligand. The principle transfers; the protein-specific
   chain convention does not.
3. **Native outputs plus normalized outputs.** Both apps should preserve native
   scientific evidence and publish portable downstream artifacts.
4. **Evidence presets.** Template/MSA presets are domain-specific, but the UI
   pattern can inform clear mn-ligand protocol presets that map visibly to actual
   engine inputs.
5. **Benchmark matrices.** Engine-by-target coverage and exact source-run
   selection could inform future docking, pocket, or prediction comparison
   matrices in mn-ligand.
6. **Capacity benchmarking.** Separate practical resource qualification from
   scientific benchmarking. mn-ligand could later use this for library size,
   receptor size, trajectory size, or engine throughput.
7. **History-based runtime estimates.** This is useful for long docking,
   cofolding, MD, and free-energy jobs when grouped by engine and workload.
8. **Candidate import with source-rank preservation.** The same principle applies
   to imported compound sets, poses, scores, and external campaign results.
9. **Detection-guided interactive selection.** Overlaying multiple hotspot
   detectors without forcing consensus is similar to mn-ligand's complementary
   pocket indicators.

## Concepts that should remain protein-design-specific

- target/binder chain naming conventions;
- binder/nonbinder labels and benchmark metrics;
- ProteinMPNN sequence-redesign stages;
- generator-specific backbone workflows;
- target MSA/template evidence modes;
- staged binder-design funnel semantics;
- PAE-indexed binder-versus-target scoring assumptions.

Share generic infrastructure patterns only. Do not copy these scientific
semantics into mn-ligand.

## Desired architectural direction

Retain the current product and candidate model, but evolve toward:

- portable run-relative typed artifact references;
- a manifest-driven engine registry;
- extend resource admission to memory and scratch space, and strengthen worker
  recovery controls;
- smaller workflow and result-renderer modules;
- reusable result frames with tool-specific extensions;
- explicit workflow parent/child state;
- first-party automated tests for contracts and pages;
- honest installation and validation diagnostics.
