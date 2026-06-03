FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG DOWNLOAD_WEIGHTS=false
ARG DOWNLOAD_TRAINING_DATA=false
ARG INSTALL_COLABFOLD=true
ARG INSTALL_BOLTZ2=true
ARG BOLTZ2_ENV_NAME=boltz2

WORKDIR /opt/genie3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        libgfortran5 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY tools_to_implement/genie3/ /opt/genie3/

RUN if [ "${INSTALL_COLABFOLD}" = "true" ]; then \
        conda install -c conda-forge -c bioconda \
            kalign2=2.04 \
            hhsuite=3.3.0 \
            mmseqs2=18.8cc5c \
            openmm \
            "python=3.10" \
            -y; \
    fi \
    && conda clean -a -y

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel "numpy>=2.0.2,<3" Cython \
    && python -m pip install --no-cache-dir --no-build-isolation -e . \
    && if [ "${INSTALL_COLABFOLD}" = "true" ]; then \
        python -m pip install --no-cache-dir "colabfold[alphafold]" \
        && python -m pip install --no-cache-dir --upgrade "jax[cuda12]==0.5.3" \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; \
       fi \
    && mkdir -p /opt/genie3/packages \
    && cd /opt/genie3/packages \
    && git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git \
    && git clone --depth 1 https://github.com/DunbrackLab/IPSAE.git \
    && if [ "${DOWNLOAD_WEIGHTS}" = "true" ] || [ "${DOWNLOAD_TRAINING_DATA}" = "true" ]; then \
        cd /opt/genie3; \
        if [ "${DOWNLOAD_WEIGHTS}" = "true" ] && [ "${DOWNLOAD_TRAINING_DATA}" = "true" ]; then \
          bash scripts/setup/download.sh; \
        elif [ "${DOWNLOAD_WEIGHTS}" = "true" ]; then \
          bash scripts/setup/download.sh --weights; \
        else \
          bash scripts/setup/download.sh --data; \
        fi; \
       fi

RUN mkdir -p /opt/genie3/packages/TMscore \
    && cd /opt/genie3/packages/TMscore \
    && curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 \
        https://zhanggroup.org/TM-score/TMscore.cpp -o TMscore.cpp \
    && g++ -O3 -ffast-math -lm -o TMscore TMscore.cpp \
    && curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 \
        https://zhanggroup.org/TM-align/TMalign.cpp -o TMalign.cpp \
    && g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp \
    && chmod +x TMscore TMalign

RUN if [ "${INSTALL_BOLTZ2}" = "true" ]; then \
        conda create -y -n "${BOLTZ2_ENV_NAME}" -c conda-forge python=3.10 pip \
        && conda run -n "${BOLTZ2_ENV_NAME}" python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
        && conda run -n "${BOLTZ2_ENV_NAME}" python -m pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu128 \
            torch torchvision torchaudio \
        && conda run -n "${BOLTZ2_ENV_NAME}" python -m pip install --no-cache-dir \
            boltz \
            cuequivariance-torch \
            cuequivariance-ops-torch-cu12 \
        && printf '#!/usr/bin/env bash\nexec conda run --no-capture-output -n %s boltz "$@"\n' "${BOLTZ2_ENV_NAME}" > /usr/local/bin/boltz \
        && chmod +x /usr/local/bin/boltz \
        && conda clean -a -y; \
    fi

RUN conda install -c conda-forge -c bioconda dssp -y \
    && mkdir -p /opt/genie3/packages/dssp-2.3.0 \
    && ln -sf /opt/conda/bin/mkdssp /opt/genie3/packages/dssp-2.3.0/mkdssp \
    && conda clean -a -y

COPY containers/genie3/entrypoint.sh /usr/local/bin/genie3-entrypoint

ENV PYTHONPATH=/opt/genie3/src:/opt/genie3/packages/ProteinMPNN:/opt/genie3/packages/IPSAE \
    XDG_CACHE_HOME=/cache/xdg-cache \
    GENIE3_COLABFOLD_DATA_DIR=/cache/xdg-cache/colabfold \
    BOLTZ_CACHE=/cache \
    PATH=/opt/genie3/packages/TMscore:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN chmod +x /usr/local/bin/genie3-entrypoint

ENTRYPOINT ["genie3-entrypoint"]
CMD ["genie3", "--help"]
