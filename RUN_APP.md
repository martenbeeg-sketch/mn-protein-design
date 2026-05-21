# Run mn-protein-design

The app runs in its own Mambaforge/Conda environment named
`mn-protein-design`.

Create or update the environment:

```bash
/home/user/mambaforge/bin/conda env update -f environment.yml --prune
```

This installs Nextflow into the same app environment. The design page can use
either direct Docker execution or the app-owned local Nextflow pipeline.

Check Nextflow:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/nextflow -version
```

Start the app:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design init
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design app --server.address 0.0.0.0
```

The app automatically chooses the first free port starting at `8501`.
To start looking from a different port:

```bash
/home/user/mambaforge/envs/mn-protein-design/bin/mn-protein-design app --server.address 0.0.0.0 --port 8508
```

Then open:

```text
http://localhost:8501
```

The default runtime directory is:

```text
./mn-protein-design-workdir
```

Detection containers used by the PPI / Hotspot Detection page:

```bash
docker compose build scannet
docker compose build surf2spot
docker compose build masif-seed
```

Sequence-design uses existing images:

```text
ovo-ligandmpnn:latest
ovoex-foundry-cu128:latest
```
