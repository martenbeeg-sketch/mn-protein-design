FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_XET=1 \
    ESMCFOLD_CCD_PATH=/ref/biohub-esm/ESMFold2/ccd.pkl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

WORKDIR /opt/esm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1

COPY tools_to_implement/esm/ /opt/esm/

RUN python -m pip install "huggingface_hub[cli]" \
    && python -m pip install -e /opt/esm

COPY containers/biohub-esm/entrypoint.sh /usr/local/bin/biohub-esm-entrypoint
COPY containers/biohub-esm/smoke_test.py /usr/local/bin/biohub-esm-smoke-test

RUN mkdir -p "$HF_HOME" \
    && chmod 777 /cache /cache/huggingface \
    && chmod +x /usr/local/bin/biohub-esm-entrypoint /usr/local/bin/biohub-esm-smoke-test

ENTRYPOINT ["biohub-esm-entrypoint"]
CMD ["biohub-esm-smoke-test"]
