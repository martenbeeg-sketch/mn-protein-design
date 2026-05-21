FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG DOWNLOAD_WEIGHTS=false
ARG DOWNLOAD_TRAINING_DATA=false

WORKDIR /opt/genie3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        g++ \
        gcc \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY tools_to_implement/genie3/ /opt/genie3/

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel "numpy>=2.0.2,<3" Cython \
    && python -m pip install --no-cache-dir --no-build-isolation -e . \
    && mkdir -p /opt/genie3/packages \
    && cd /opt/genie3/packages \
    && git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git \
    && git clone --depth 1 https://github.com/DunbrackLab/IPSAE.git \
    && mkdir -p TMscore \
    && cd TMscore \
    && wget -q https://zhanggroup.org/TM-score/TMscore.cpp \
    && g++ -O3 -ffast-math -lm -o TMscore TMscore.cpp \
    && wget -q https://zhanggroup.org/TM-align/TMalign.cpp \
    && g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp \
    && chmod +x TMscore TMalign \
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

COPY containers/genie3/entrypoint.sh /usr/local/bin/genie3-entrypoint

ENV PYTHONPATH=/opt/genie3/src:/opt/genie3/packages/ProteinMPNN:/opt/genie3/packages/IPSAE \
    PATH=/opt/genie3/packages/TMscore:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN chmod +x /usr/local/bin/genie3-entrypoint

ENTRYPOINT ["genie3-entrypoint"]
CMD ["genie3", "--help"]
