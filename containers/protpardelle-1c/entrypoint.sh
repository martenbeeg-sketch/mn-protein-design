#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${PROTPARDELLE_OUTPUT_DIR:-/work/output}"

exec "$@"
