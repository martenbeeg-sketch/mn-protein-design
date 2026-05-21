FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/PXDesign

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY tools_to_implement/PXDesign/ /opt/PXDesign/

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        PyYaml \
        numpy==1.26.3 \
        scipy \
        ml_collections \
        tqdm \
        pandas \
        optree \
        rdkit \
        biopython==1.83 \
        modelcif==0.7 \
        biotite==1.0.1 \
        scikit-learn \
        scikit-learn-extra \
        "deepspeed>=0.15.1" \
        protobuf==3.20.2 \
        icecream \
        ipdb \
        wandb \
        matplotlib==3.9.2 \
        ipywidgets \
        py3Dmol \
        fair-esm \
        posix_ipc \
        einops \
        transformers==4.51.3 \
        "jax[cuda12]==0.4.29" \
        optax==0.2.5 \
        dm-haiku==0.0.13 \
        natsort \
        dm-tree \
    && python -m pip install --no-cache-dir git+https://github.com/bytedance/Protenix.git@v0.5.0+pxd --no-deps \
    && python -m pip install --no-cache-dir git+https://github.com/sokrypton/ColabDesign.git --no-deps \
    && python -m pip install --no-cache-dir git+https://github.com/bytedance/PXDesignBench.git@v0.1.2 --no-deps \
    && python -m pip install --no-cache-dir -e .

RUN git clone -b v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass

COPY containers/pxdesign/entrypoint.sh /usr/local/bin/pxdesign-entrypoint

ENV CUTLASS_PATH=/opt/cutlass \
    PROTENIX_DATA_ROOT_DIR=/ref/pxdesign/release_data/ccd_cache \
    TOOL_WEIGHTS_ROOT=/ref/pxdesign/tool_weights \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN chmod +x /usr/local/bin/pxdesign-entrypoint

ENTRYPOINT ["pxdesign-entrypoint"]
CMD ["pxdesign", "--help"]
