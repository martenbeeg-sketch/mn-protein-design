# BindCraft 2 container

The app uses a separate BC2 image so the existing BindCraft 1.5 image and workflow remain available.
The build script checks out the pinned upstream revision `4a56313e96afc13c443d88427281cf2169a1c9ca`
and builds its CUDA 13 Dockerfile as `mn-bindcraft2:4a56313-cu13`.

```bash
../mn-tool-containers/scripts/build-bindcraft2.sh
```

The image includes BC2 and its shipped ProteinMPNN weights. AlphaFold parameters stay outside the
image and are mounted read-only from the configured app reference root at `/ref/alphafold_models`.
The app passes that directory to BC2 as `BINDCRAFT_AF2_PARAMS`. This keeps large model files
separate from portable job folders and lets the app use its normal reference-root mapping.

BC2 needs an NVIDIA GPU and a Linux host driver that supports CUDA 13. Check that JAX reaches a
GPU before submitting a long campaign:

```bash
docker run --rm --gpus device=0 --entrypoint python3 mn-bindcraft2:4a56313-cu13 \
  -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
```

The image performs its package and shipped-weight self-check during the build. The command above is
the runtime GPU check; it should be repeated on an RTX 5090 host before relying on that hardware.

Build a local image for another source revision only by deliberately changing the pinned commit,
image tag, and manifest together. Keep the upstream `LICENSE` with any redistributed source or
image, and follow its hosted-service restriction.
