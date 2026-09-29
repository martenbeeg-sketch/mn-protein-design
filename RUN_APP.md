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
docker compose build pesto
docker compose build surf2spot
docker compose build masif-seed
```

PeSTo keeps its model parameters in the shared reference folder rather than in
the image. On another computer, copy the checkpoint from a trusted source or
from your existing reference data to this location before running PeSTo:

```bash
sudo install -d /mnt/db/reference_files/pesto/i_v4_1
test -s /mnt/db/reference_files/pesto/i_v4_1/model_ckpt.pt
```

Sequence-design uses existing images:

```text
mn-ligandmpnn:latest
mn-foundry:cu128
```
