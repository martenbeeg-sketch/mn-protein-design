FROM condaforge/mambaforge:23.11.0-0

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/ScanNet

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        git \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY tools_to_implement/ScanNet/requirements.txt /tmp/scannet-requirements.txt

# ScanNet is a TensorFlow 1.x application. TF 1.14 cannot use RTX 50xx/Blackwell
# GPUs, so this image intentionally provides a CPU-stable runtime.
RUN mamba create -y -n scannet python=3.6.13 pip \
    && mamba run -n scannet pip install --no-cache-dir --upgrade "pip<22" setuptools wheel \
    && mamba run -n scannet pip install --no-cache-dir -r /tmp/scannet-requirements.txt \
    && mamba clean -afy

COPY tools_to_implement/ScanNet/ /opt/ScanNet/
COPY containers/scannet/entrypoint.sh /usr/local/bin/scannet-entrypoint

ENV PATH=/opt/conda/envs/scannet/bin:$PATH \
    PYTHONPATH=/opt/ScanNet \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=""

RUN chmod +x /usr/local/bin/scannet-entrypoint

ENTRYPOINT ["scannet-entrypoint"]
CMD ["python", "predict_bindingsites.py", "--help"]
