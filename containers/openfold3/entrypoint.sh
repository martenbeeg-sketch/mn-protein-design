#!/usr/bin/env bash
set -eo pipefail

source /opt/activate.sh

mkdir -p "${OPENFOLD_CACHE:-/ref/openfold3}"

exec "$@"
