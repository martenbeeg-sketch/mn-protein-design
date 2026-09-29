# Install and migrate workdir data

This workflow installs the app code and Python/Docker environment separately
from its results, then copies the complete app `workdir` to a new app-home. The
source installation stays usable: export writes a second copy and never moves,
deletes, or rewrites source data.

The bundle includes the workdir tree, configured run folders, job inputs,
metadata, results, targets, candidate data, and related files. It rewrites
managed paths in the exported copy so runs and workdir files resolve from the
new app-home. `reference_files`, model weights/databases, Docker images,
temporary files outside the workdir, and source code are handled separately.

## 1. Install the app on the destination

Install the repository and Python dependencies on the other computer. Build or
transfer the Docker images there as needed. Do not run `mn-protein-design init`
for the destination path before importing; import requires a new app-home and
will not merge with or overwrite existing files.

## 2. Audit and export on the source

Let queued, running, and paused jobs finish. Check the effective paths and size:

```bash
mn-protein-design portability audit
```

Create the export on a different directory or filesystem with sufficient free
space. The destination directory must not already exist:

```bash
EXPORT_DIR=/path/with/free-space/mn-protein-design-portable-$(date +%F)
mn-protein-design portability export "$EXPORT_DIR"
mn-protein-design portability verify "$EXPORT_DIR"
du -sh "$EXPORT_DIR"
```

Export refuses to run while jobs are queued, running, preparing, or paused. It
also refuses to create a bundle if operational paths point outside the copied
workdir/runs or configured reference root. Run `portability audit --json` to
inspect those paths. The export includes a SHA-256 and size record for every
file; verification checks those records and all managed paths.

## 3. Transfer a copy

Copy the verified export directory to the destination computer. `rsync` can
resume an interrupted large transfer:

```bash
rsync -a --partial --append-verify --info=progress2 \
  "$EXPORT_DIR/" \
  user@NEW_HOST:/path/to/mn-protein-design-portable/
```

This copies the bundle. It does not remove the source workdir or export.

## 4. Import on the destination

Verify the transferred files, then import into a new app-home:

```bash
mn-protein-design portability verify /path/to/mn-protein-design-portable
mn-protein-design portability import \
  /path/to/mn-protein-design-portable \
  --app-home /path/to/mn-protein-design-data
```

Import verifies checksums again, copies the bundle to a staging directory, and
publishes it only after the copy is complete. It refuses an existing app-home.
The destination contains the imported workdir and results; the source computer
continues using its original workdir unchanged.

## 5. Configure references and open the imported results

Copy or mount the reference files separately, then point the app to the
destination paths printed by `portability import`:

```bash
mn-protein-design app \
  --app-home /path/to/mn-protein-design-data \
  --runs-dir /path/to/mn-protein-design-data/workdir/runs \
  --reference-dir /path/to/copied/reference_files
```

The Jobs and Results pages read from the imported runs directory. Managed
`runs:///` and `app:///` paths resolve against this app-home. `reference:///`
paths resolve against `--reference-dir`; reference files are not inside the
portable bundle.

After checking that the imported data opens correctly, install activation-free
terminal and desktop launchers from the app environment:

```bash
mn-protein-design install-launchers \
  --app-home /path/to/mn-protein-design-data \
  --runs-dir /path/to/mn-protein-design-data/workdir/runs \
  --reference-dir /path/to/copied/reference_files
```

Then run `mn-protein-design-app` or open the **MN Protein Design** desktop
shortcut. The launcher stores these selected paths; rerun the installer if you
move the imported data or Python environment.

New job input/result metadata and candidate path references are stored as
run-relative paths or managed URIs. Existing jobs with absolute paths are
normalized inside the export copy when their files are under the transferred
workdir/runs or the separately configured reference root.

If the source had both the default `workdir/runs` and a separate configured
runs directory, export merges them into the imported `workdir/runs`. It checks
for conflicting files and refuses to overwrite different data. Use the runs
path printed by import when launching the app.

## Safety and limits

- Existing source files are only read. Portable path rewriting happens in the
  export copy.
- Historical job directories remain unchanged on the source computer.
- Import refuses to modify an existing destination app-home. Export merges the
  source's configured and legacy run folders only when duplicate files match;
  conflicts stop the export.
- Worker PID/lock state is not copied; no worker is started on import.
- Reference files, model databases/weights, Docker images, Python environments,
  secrets, and external software remain separate installation steps.
- Absolute operational paths outside managed workdir/runs/reference roots stop
  export, so a bundle will not silently claim to be portable when it is missing
  a required file.
