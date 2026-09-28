ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE} AS builder

ARG PIXI_VERSION=v0.68.0
ARG PIXI_ENV=openfold3-cuda13
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libaio-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /opt/pixi-project
COPY tools_to_implement/openfold-3/pixi.toml tools_to_implement/openfold-3/pixi.lock tools_to_implement/openfold-3/pyproject.toml ./
COPY tools_to_implement/openfold-3/openfold3/__init__.py ./openfold3/

RUN CUDA_FEATURE=$(echo "${PIXI_ENV}" | grep -oP 'cuda\d+') \
    && if [ -n "${CUDA_FEATURE}" ]; then \
        CUDA_VERSION=$(grep -A1 "\[feature\.${CUDA_FEATURE}\.system-requirements\]" pixi.toml \
            | grep -oP 'cuda\s*=\s*"\K[^"]+'); \
        export CONDA_OVERRIDE_CUDA="${CUDA_VERSION}"; \
    fi \
    && pixi install --frozen -e "${PIXI_ENV}"

FROM ${BASE_IMAGE}

ARG PIXI_VERSION=v0.68.0
ARG PIXI_ENV=openfold3-cuda13
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libaio-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /opt/pixi-project
COPY tools_to_implement/openfold-3/pixi.toml tools_to_implement/openfold-3/pixi.lock tools_to_implement/openfold-3/pyproject.toml ./
COPY --from=builder /opt/pixi-project/.pixi/envs/${PIXI_ENV} ./.pixi/envs/${PIXI_ENV}

RUN mkdir -p ./.pixi/envs/${PIXI_ENV}/caches/triton \
             ./.pixi/envs/${PIXI_ENV}/caches/torch_extensions

RUN CUDA_FEATURE=$(echo "${PIXI_ENV}" | grep -oP 'cuda\d+') \
    && if [ -n "${CUDA_FEATURE}" ]; then \
        CUDA_VERSION=$(grep -A1 "\[feature\.${CUDA_FEATURE}\.system-requirements\]" pixi.toml \
            | grep -oP 'cuda\s*=\s*"\K[^"]+'); \
        export CONDA_OVERRIDE_CUDA="${CUDA_VERSION}"; \
    fi \
    && pixi shell-hook -e "${PIXI_ENV}" -s bash > /opt/activate.sh

WORKDIR /opt/openfold3
COPY tools_to_implement/openfold-3/ /opt/openfold3/

RUN bash -c "source /opt/activate.sh && pip install --no-deps --editable ."

COPY containers/openfold3/entrypoint.sh /usr/local/bin/openfold3-entrypoint

ENV KMP_AFFINITY=none \
    OPENFOLD_CACHE=/ref/openfold3 \
    TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0+PTX" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN chmod +x /usr/local/bin/openfold3-entrypoint

ENTRYPOINT ["openfold3-entrypoint"]
CMD ["run_openfold", "--help"]
