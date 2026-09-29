# mn-protein-design

Local Streamlit workbench for Docker-orchestrated protein target preparation,
binder design, sequence optimization, refolding, analysis, and benchmarking.

## Development handoff

New development sessions should read:

1. `APP_DEVELOPMENT.md`
2. `APP_feature.md`
3. `APP_MISSING.md`
4. this README

Detailed current features are in `APP_FEATURES.md`. Engine integration guidance
is in `LLM_ENGINE_INTEGRATION_GUIDE.md`, and native container checks are in
`smoke_tests/README.md`.

## Install

Clone `mn-compute-scheduler` beside this repository first; both apps install
that shared resource manager from `../mn-compute-scheduler`.

Create or update the documented Conda environment:

```bash
/home/user/mambaforge/bin/conda env update -f environment.yml --prune
```

Install the package if needed:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pip install -e ../mn-compute-scheduler
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pip install -e .
```

## Run

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design init
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design app \
  --server.address 0.0.0.0
```

The app selects the first free port beginning at 8501. To choose a different
starting port:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design app \
  --server.address 0.0.0.0 \
  --port 8508
```

To run it without activating the Conda environment, install per-user launchers
once from the app environment:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design install-launchers
```

This creates `~/.local/bin/mn-protein-design` for CLI commands,
`~/.local/bin/mn-protein-design-app` as a one-command app launcher, and a
**MN Protein Design** shortcut on `~/Desktop`. The app launcher pins the
effective app-home, runs, reference, and temporary paths at install time, so it
opens the same data. Run the installer again if you move the Python environment
or change those paths. Use `--no-desktop` to omit the desktop shortcut.

## Background jobs and resources

Queued workflows use a separate local worker service. The app starts it
automatically when a queued job is submitted, and it continues running if you
close the browser or stop Streamlit. To keep the service available before
submitting jobs, run it in a terminal:

```bash
mn-protein-design worker
```

The default worker capacity is up to 32 CPU slots. Jobs request four CPU slots
by default; saved thread settings such as `pyrosetta_nprocs` can request more.
CPU-only jobs reserve slots, and GPU jobs reserve both slots and their selected
device. Jobs inside a workflow share the parent's reservation. Docker receives
matching thread settings and uses `--cpus` when the host supports cgroup CPU
quotas. Jobs that cannot start show a wait reason in the Jobs table.

Set `MN_PROTEIN_DESIGN_WORKER_CPU_SLOTS` before starting the app or worker to
change the default capacity. Check or stop the queue service with:

```bash
mn-protein-design worker-status
mn-protein-design worker-stop
```

Stopping the service prevents it from dispatching new jobs; already running
jobs continue. The service does not automatically start after a machine reboot.

For scripted local campaign submission, individual workflow runs, and job
inspection, see [CLI_AUTOMATION.md](CLI_AUTOMATION.md). The automation CLI
creates normal app jobs and writes outputs to the configured run directory.

## Runtime paths

Current defaults:

```text
app home:  <checkout>/mn-protein-design-workdir
runs:      <app-home>/workdir/runs
temporary: <checkout>/.tmp
references:/mnt/db/reference_files when available
```

Overrides:

```bash
export MN_PROTEIN_DESIGN_APP_HOME=/path/to/app-home
export MN_PROTEIN_DESIGN_RUN_DIR=/path/to/runs
export MN_PROTEIN_DESIGN_REFERENCE_DIR=/path/to/reference_files
export TMPDIR=/path/to/tmp
```

`Settings` shows effective runtime, reference, executable, and image paths. Some
workflow adapters still contain direct shared-reference or checkout assumptions;
see `APP_MISSING.md`.

New job JSON, candidate structure, and artifact records use run-relative paths
or managed references (`runs:///`, `reference:///`, and `app:///`) where
possible. The Settings page shows how those prefixes resolve on this computer.
Existing absolute paths remain readable, with relocation support for selected
legacy run and reference paths. To copy the complete workdir to another computer
without changing this machine's data, see [INSTALL_AND_MIGRATION.md](INSTALL_AND_MIGRATION.md)
or use the `mn-protein-design portability` CLI commands.

## Main workflows

- Target Preparation
- PPI / Hotspot Detection
- Target Cropping
- Design
- Design Campaigns
- Sequence Design
- Candidate Sets
- Refolding / Validation
- Analysis
- Binder Benchmark
- Capacity Benchmark

Reusable scientific outputs are normalized under:

```text
artifacts/normalized_candidates/candidates.jsonl
```

New engine integrations must preserve explicit target/binder chain roles and
native outputs. See `LLM_ENGINE_INTEGRATION_GUIDE.md`.

## Docker and references

Scientific engines run in separate Docker images. Large model weights, MSA
databases, and caches belong under the configured reference root, normally
`/mnt/db/reference_files`.

Container build and setup notes:

```text
containers/README.md
containers/<engine>/README.md
smoke_tests/README.md
```

Build the configured images from the shared container repository with:

```bash
../mn-tool-containers/build.sh mn-protein-design
```

Run only the engine-specific native smoke needed for a change; many checks
require substantial model downloads and GPU time.

## Tests

Install the development extra in the app environment:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pip install -e '.[dev]'
```

Run the first-party CPU/local suite with:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/python -m pytest -q
```

These tests do not require Docker, a GPU, model weights, or network access.
Engine-native Docker/GPU smoke tests remain separate. Tests inside
`tools_to_implement/` and `ui_inspiration/` belong to upstream/reference projects,
not this app's regression suite.

## Repository caution

Keep simulation outputs, uploaded structures, model weights, caches, and other
user-specific runtime data out of source commits. Preserve historical run
folders and local source/reference data.
