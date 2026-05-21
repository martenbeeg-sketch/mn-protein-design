FROM pablogainza/masif:latest AS masif_deps

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_ESMFOLD=false

WORKDIR /opt/Surf2Spot

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apbs \
        bash \
        ca-certificates \
        dssp \
        g++ \
        gcc \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libxrender1 \
        libxt6 \
        pdb2pqr \
        pymol \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY tools_to_implement/Surf2Spot/ /opt/Surf2Spot/
COPY --from=masif_deps /root/msms/msms.x86_64Linux2.2.6.1 /usr/local/bin/msms.x86_64Linux2.2.6.1
COPY --from=masif_deps /usr/local/share/apbs/tools/bin/multivalue /usr/local/bin/multivalue

# Surf2Spot hardcodes these MaSIF dependency paths in
# Surf2Spot/data/generate_surface.py. Build the expected tree into the image so
# jobs can run without entrypoint-time mutation.
RUN mkdir -p \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/APBS-3.4.1/bin \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/APBS-3.4.1/share/apbs/tools/bin \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/pdb2pqr \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/msms \
    && ln -sf "$(command -v apbs)" \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/APBS-3.4.1/bin/apbs \
    && ln -sf "$(command -v multivalue)" \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/APBS-3.4.1/share/apbs/tools/bin/multivalue \
    && ln -sf "$(command -v pdb2pqr)" \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/pdb2pqr/pdb2pqr \
    && cp /usr/local/bin/msms.x86_64Linux2.2.6.1 \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/msms/msms.x86_64Linux2.2.6.1 \
    && chmod +x \
        /usr/local/bin/msms.x86_64Linux2.2.6.1 \
        /usr/local/bin/multivalue \
        /opt/Surf2Spot/Surf2Spot/data/masif/extra_dependence/msms/msms.x86_64Linux2.2.6.1 \
    && ln -sf /usr/local/bin/msms.x86_64Linux2.2.6.1 /usr/local/bin/msms

# Install PyTorch Geometric wheels matched to PyTorch 2.7 / CUDA 12.8 for
# Blackwell-class GPUs. Surf2Spot's checked-in requirements include older
# Torch/Triton pins, so install the project without dependency resolution and
# then install only the runtime dependencies it actually imports.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f https://data.pyg.org/whl/torch-2.7.0+cu128.html \
    && python -m pip install --no-cache-dir \
        aiohttp \
        biopandas \
        biopython==1.81 \
        h5py \
        huggingface-hub \
        ipython \
        joblib \
        matplotlib \
        numba \
        numpy==1.24.3 \
        opencv-python-headless \
        pandas==2.0.1 \
        pillow \
        plyfile \
        psutil \
        pydantic==1.10.8 \
        PyYAML \
        requests \
        rdkit \
        safetensors \
        scikit-learn==1.3.2 \
        scipy==1.10.1 \
        sentencepiece \
        tokenizers==0.19.1 \
        torch-geometric==2.6.1 \
        tqdm \
        transformers==4.41.2 \
        typer \
    && python -m pip install --no-cache-dir --no-deps pymol-open-source \
    && python -m pip install --no-cache-dir --no-deps -e . \
    && if [ "${INSTALL_ESMFOLD}" = "true" ]; then \
        python -m pip install --no-cache-dir "fair-esm[esmfold]" \
          'dllogger @ git+https://github.com/NVIDIA/dllogger.git' \
          'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'; \
       fi

COPY containers/surf2spot/entrypoint.sh /usr/local/bin/surf2spot-entrypoint

ENV PYTHONPATH=/opt/Surf2Spot \
    PATH=/opt/Surf2Spot/Surf2Spot/data/feature_extraction:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN chmod +x /usr/local/bin/surf2spot-entrypoint
RUN chmod +x /opt/Surf2Spot/Surf2Spot/data/chainsaw/stride/stride || true

ENTRYPOINT ["surf2spot-entrypoint"]
CMD ["Surf2Spot", "--help"]
