ARG BASE_IMAGE=ghcr.io/sokrypton/colabfold:1.6.1-cuda12
ARG JAX_CUDA_SPEC=jax[cuda12]<0.8

FROM ${BASE_IMAGE}

ARG JAX_CUDA_SPEC=jax[cuda12]<0.8

USER root

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/cache/matplotlib \
    XDG_CACHE_HOME=/cache/xdg-cache \
    COLABFOLD_DATA=/cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /cache /work

COPY containers/colabfold/entrypoint.sh /usr/local/bin/colabfold-entrypoint
COPY containers/colabfold/gpu-smoke-test.py /usr/local/bin/colabfold-gpu-smoke-test
RUN python -m pip install --no-cache-dir --upgrade "${JAX_CUDA_SPEC}" \
    && chmod +x /usr/local/bin/colabfold-entrypoint /usr/local/bin/colabfold-gpu-smoke-test \
    && python - <<'PY'
import jax
print("jax", jax.__version__)
PY

WORKDIR /work
VOLUME ["/cache", "/work"]

ENTRYPOINT ["/usr/bin/tini", "--", "colabfold-entrypoint"]
CMD ["colabfold_batch", "--help"]
