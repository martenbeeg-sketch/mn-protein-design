#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/ref/biohub-esm/hf-cache}"
export ESMCFOLD_CCD_PATH="${ESMCFOLD_CCD_PATH:-/ref/biohub-esm/ESMFold2/ccd.pkl}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0;12.0}"

exec "$@"
