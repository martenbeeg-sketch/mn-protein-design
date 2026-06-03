#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${XDG_CACHE_HOME:-/cache/xdg-cache}" "${MPLCONFIGDIR:-/cache/matplotlib}" /cache /work

if [[ "$#" -eq 0 ]]; then
  set -- colabfold_batch --help
fi

exec "$@"
