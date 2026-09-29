# ColabFold container

This image wraps the upstream ColabFold CUDA 12 container so the app can use a
stable local tag and the shared model cache at `/mnt/db/reference_files`.

The upstream Docker guide recommends CUDA 12 images, NVIDIA Docker, and a
persistent cache mount for AlphaFold2 weights. The current upstream container
registry publishes `ghcr.io/sokrypton/colabfold:1.6.1-cuda12`.

The Dockerfile also reinstalls the current `jax[cuda12]<0.8` wheel stack at
build time. This keeps the image compatible with the ColabFold dependency range
while avoiding stale JAX CUDA wheels. That matters for RTX 50-series cards:
Blackwell / RTX 5090 needs a recent driver and CUDA/JAX stack that can run
`sm_120` kernels or PTX JIT fallback.

## Build

```bash
docker build \
  -f containers/colabfold/Dockerfile \
  -t mn-colabfold:1.6.1-cu12 \
  .
```

To override the JAX CUDA wheel selection:

```bash
docker build \
  -f containers/colabfold/Dockerfile \
  --build-arg 'JAX_CUDA_SPEC=jax[cuda12]<0.8' \
  -t mn-colabfold:1.6.1-cu12 \
  .
```

## Check the CLI

```bash
docker run --rm --gpus all \
  -v /mnt/db/reference_files/alphafold_models:/cache/params:rw \
  -v "$PWD":/work:rw \
  -w /work \
  mn-colabfold:1.6.1-cu12 \
  colabfold_batch --help --data /cache
```

## GPU smoke test

Run this on both the local RTX 4090 and the future RTX 5090 host:

```bash
docker run --rm --gpus all \
  -v /mnt/db/reference_files/alphafold_models:/cache/params:rw \
  mn-colabfold:1.6.1-cu12 \
  colabfold-gpu-smoke-test
```

For RTX 5090, the important part is that JAX reports a GPU backend and the
matrix multiply completes. If it falls back to CPU or reports unsupported
`sm_120`, rebuild with newer JAX wheels and make sure the host NVIDIA driver is
new enough for CUDA 12.8+/Blackwell.

## Download or refresh weights

Use this only if `/mnt/db/reference_files/alphafold_models` does not already
contain the ColabFold/AlphaFold2 weights.

```bash
docker run --rm --gpus all \
  -v /mnt/db/reference_files/alphafold_models:/cache/params:rw \
  mn-colabfold:1.6.1-cu12 \
  python -m colabfold.download /cache
```

## Run prediction from prepared A3M inputs

The de novo binder scoring workflow already creates ColabFold `.a3m` files in
`ColabFold/input_folder`. If those inputs were produced with AlphaFast/AF3 data
pipeline MSAs or another trusted MSA generator, skip ColabFold MSA generation
and run prediction directly:

```bash
docker run --rm --gpus all --shm-size=32G \
  -v /mnt/db/reference_files/alphafold_models:/cache/params:rw \
  -v "$PWD":/work:rw \
  -w /work \
  mn-colabfold:1.6.1-cu12 \
  colabfold_batch \
    /work/ColabFold/input_folder \
    /work/ColabFold/ptm_output \
    --data /cache \
    --calc-extra-ptm \
    --num-recycle 3 \
    --num-models 3
```

## Generate MSAs with ColabFold

For small target-only jobs, ColabFold can still make the MSA first:

```bash
docker run --rm --gpus all --shm-size=32G \
  -v /mnt/db/reference_files/alphafold_models:/cache/params:rw \
  -v "$PWD":/work:rw \
  -w /work \
  mn-colabfold:1.6.1-cu12 \
  colabfold_batch /work/unique_msa /work/unique_msa/msa --msa-only --data /cache
```

For larger benchmark runs, prefer the app's local AlphaFast/AF3 data pipeline or
a local MMseqs database rather than the public ColabFold API.
