FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/protpardelle-1c

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY tools_to_implement/protpardelle-1c/ /opt/protpardelle-1c/

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "huggingface_hub[cli]" \
    && python -m pip install --no-cache-dir -e .

COPY containers/protpardelle-1c/entrypoint.sh /usr/local/bin/protpardelle-entrypoint

ENV PROJECT_ROOT_DIR=/opt/protpardelle-1c \
    PROTPARDELLE_MODEL_PARAMS=/ref/protpardelle-1c/model_params \
    ESMFOLD_PATH=/ref/protpardelle-1c/model_params/ESMFold \
    PROTEINMPNN_WEIGHTS=/ref/protpardelle-1c/model_params/ProteinMPNN/vanilla_model_weights \
    LIGANDMPNN_WEIGHTS=/ref/protpardelle-1c/model_params/LigandMPNN \
    PROTPARDELLE_OUTPUT_DIR=/work/output \
    HF_HOME=/ref/protpardelle-1c/hf-cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

RUN chmod +x /usr/local/bin/protpardelle-entrypoint

ENTRYPOINT ["protpardelle-entrypoint"]
CMD ["python", "-m", "protpardelle.sample", "--help"]
