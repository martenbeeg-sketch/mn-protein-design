FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/Protenix

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        hmmer \
        kalign \
        libc6-dev \
        make \
        postgresql-client \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY tools_to_implement/Protenix/ /opt/Protenix/

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --extra-index-url https://pypi.org/simple -r requirements.txt \
    && python -m pip install --no-cache-dir -e .

RUN git clone -b v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass

ENV CUTLASS_PATH=/opt/cutlass \
    PROTENIX_ROOT_DIR=/ref/protenix \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN protenix --help >/tmp/protenix-help.txt \
    && rm -f /tmp/protenix-help.txt

CMD ["protenix", "--help"]
