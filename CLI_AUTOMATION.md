# Local automation CLI

The CLI is the supported local automation interface for this app. It invokes
the same campaign workflow, file-backed job store, worker scheduler, and
normalized candidate artifacts used by Streamlit. It does not require the UI
to be open. There is no network service to configure.

## Submit a campaign

Save a request as JSON. Request schema version `1` is required. Relative target
paths resolve from the JSON file's directory; managed `app:///`, `reference:///`,
and `runs:///` paths also work. `target_chains` and hotspot residue numbers are
checked against the supplied PDB before a job is created.

```json
{
  "schema_version": 1,
  "campaign_name": "PDL1 vanilla comparison",
  "target_pdb": "app:///workdir/targets/reference_examples/bindcraft1/PDL1.pdb",
  "target_chains": ["A"],
  "binder_length": "60-100",
  "hotspots": ["A56"],
  "design_attempts": 10,
  "sequences_per_backbone": 1,
  "random_seed": 1,
  "engines": ["bindcraft", "bindcraft2"],
  "workflow_recipe": "vanilla",
  "gpu_device": "0",
  "engine_configs": {
    "bindcraft": {
      "advanced_settings_file": "default_4stage_multimer_mpnn.json",
      "filter_settings": "default_filters.json"
    },
    "bindcraft2": {
      "modality": ["binder"],
      "max_trajectories": 10,
      "number_of_final_designs": 10,
      "workers_per_gpu": "auto",
      "cpu_cores": 4
    }
  }
}
```

The example intentionally sets the native BC1 filters and the BC2 trajectory
cap. Engine options are passed to the existing workflow adapters, so settings
specific to an engine remain in that engine's `engine_configs` object. Omitted
engine options use the workflow's existing defaults. `design_attempts` is the
shared attempt count; BindCraft 2 can additionally set its own
`max_trajectories` cap.

Validate without creating a job, then submit:

```bash
mn-protein-design campaign validate --request campaign.json
mn-protein-design campaign submit --request campaign.json
```

Submit prints a JSON job reference with the short `job_code`, full `run_id`,
task group, status, and result directory. The job is queued through the same
local worker path as a UI submission and continues when Streamlit is closed.
Runtime locations can be configured with `MN_PROTEIN_DESIGN_APP_HOME`,
`MN_PROTEIN_DESIGN_RUN_DIR`, `MN_PROTEIN_DESIGN_REFERENCE_DIR`, and `TMPDIR`;
see [README.md](README.md#runtime-paths).

## Inspect and control jobs

All job commands print JSON, which makes the output straightforward to consume
from shell scripts:

```bash
mn-protein-design jobs list --status running
mn-protein-design jobs show ABCDE
mn-protein-design jobs wait ABCDE --timeout 7200 --poll-seconds 5
mn-protein-design jobs results ABCDE
mn-protein-design jobs results ABCDE --include-candidates
mn-protein-design jobs cancel ABCDE --reason "Operator requested stop"
```

Job references can be a short job code, full run ID, or
`task-group/run-ID`. If a short code is ambiguous, use the full ID. `jobs
results` includes the result JSON, candidate count, and normalized campaign
result path. `--include-candidates` adds all normalized candidate records to
the output; for large jobs, read the reported
`artifacts/normalized_candidates/candidates.jsonl` file directly instead.
Internal child jobs are hidden from the default list just as they are in the
app. Use `jobs list --include-hidden` followed by `jobs show --include-hidden
CHILD_ID` or `jobs results --include-hidden CHILD_ID` to inspect one. Cancel the
visible parent campaign so the coordinator and its children stop together.

Exit codes for `jobs wait` are `0` for successful completion, `1` for failed
or cancelled jobs, and `2` when the timeout expires. `jobs cancel` marks an
active job cancelled and preserves its run directory and partial outputs.

The Python service boundary used by the CLI is
`mn_protein_design.services.local_automation`. Its request schema is explicitly
versioned so local scripts can validate and submit through the same contract
without importing Streamlit page code. The supported service calls are
`load_campaign_request`, `submit_design_campaign_request`, `list_jobs`,
`inspect_job`, `job_results`, `wait_for_job`, and `cancel_job`.

```python
from mn_protein_design.services.local_automation import (
    submit_design_campaign_request,
    wait_for_job,
    job_results,
)

job = submit_design_campaign_request("campaign.json")
finished = wait_for_job(job["run_id"], timeout_seconds=7200)
if finished is None or finished["status"] != "completed" or finished.get("success") is not True:
    raise RuntimeError(f"Campaign did not finish successfully: {finished}")
results = job_results(job["run_id"])
print(results["candidate_count"], results["campaign_result_path"])
```

## Run individual app workflows

Use `workflow` for individual target-preparation, detection, design, sequence,
refolding, analysis, candidate-import, and benchmark operations. Queued
workflows use the same detached worker queue and run folders as the app. Small
local utilities such as masking a target or importing a candidate table finish
in the CLI process and return their completed run folder. `workflow list`
prints the complete allowlist; `workflow schema ID` prints that workflow's
parameter types, required fields, and defaults.

```bash
mn-protein-design workflow list
mn-protein-design workflow list --category refolding
mn-protein-design workflow schema sequence.ligandmpnn
```

Each request is a JSON object with `schema_version: 1`, a workflow ID, and its
parameters. Paths may be absolute, relative to the request JSON file, or use
the managed `app:///`, `reference:///`, and `runs:///` path forms. Use
`source_job` on workflows that consume normalized candidates to resolve that
job's run folder and `artifacts/normalized_candidates/candidates.jsonl`.
Alternatively, pass `source_run_dir` and `candidates_jsonl` explicitly.

For example, design sequences for all normalized candidates from an existing
job:

```json
{
  "schema_version": 1,
  "workflow": "sequence.ligandmpnn",
  "source_job": "ABCDE",
  "parameters": {
    "model_type": "ligand_mpnn",
    "num_seq_per_target": 4,
    "sampling_temp": 0.1,
    "omit_aas": "CX"
  },
  "resources": {
    "cpu_cores": 4,
    "gpu_device": "0"
  }
}
```

Validate the request before submission. Validation resolves and checks input
paths and resource choices without creating a job. Submission prints the new
job reference; follow it with the `jobs` commands above.

```bash
mn-protein-design workflow validate --request sequence-design.json
mn-protein-design workflow submit --request sequence-design.json
mn-protein-design jobs show ABCDE
```

Resource values are applied to the app's scheduler. Conflicting resource and
workflow parameter values are rejected. When resource settings are omitted,
the workflow and worker defaults apply. For target refolding and candidate
refolding, engine-specific settings are grouped under `parameters.options`;
the workflow schema lists the common supported refolding options. For
`sequence.pipeline`, pass `backend`, `design_kwargs`, and optionally
`validation_kwargs` to create one queued job with a durable continuation.
`campaign.validation_sequence` accepts a `source_job` plus an ordered `steps`
array for sequential sequence design, monomer/complex refolding, and analysis.

The workflow allowlist currently includes:

- `target.*`: preparation, cropping, masking, target candidate-set creation,
  and target refolding evaluation.
- `detection.*`: ScanNet, PeSTo, Surf2Spot, and MaSIF-seed.
- `design.*`: RFdiffusion classic, BindCraft 1 and 2, RFdiffusion3/Foundry,
  BoltzGen, PXDesign, Genie3, ESMFold2 binder design/screening,
  Protpardelle-1c, and Proteina-Complexa.
- `sequence.*`: LigandMPNN, Foundry MPNN, and the sequence/refolding pipeline.
- `refolding.*`: the app's monomer and complex refolding backends.
- `analysis.*` and `import.*`: normalized-candidate ranking plus BindCraft and
  CSV/TSV/Excel candidate imports.
- `benchmark.*` and `campaign.*`: dataset scoring, candidate refolding,
  precomputed metrics, ESMFold2 benchmarking, capacity runs, result collections,
  matrix workspaces, and validation lineages.

The CLI accepts file paths in place of browser uploads. For table imports, set
`table_path` and the column names under `parameters`. For dataset scoring, pass
an input CSV/ZIP path or an existing candidate set. Structured UI values such
as target-chain selections and Mol* residue selections can be written directly
as JSON parameter values. Browser-only previews and live viewer interaction are
not needed for queued workflow execution.

The workflow request service is `mn_protein_design.services.workflow_automation`;
its `workflow_catalog`, `workflow_schema`, `validate_workflow_request`, and
`submit_workflow_request` calls mirror the CLI. `campaign submit` remains the
higher-level interface for multi-engine design campaigns; a `workflow` request
starts one allowlisted workflow operation at a time.
