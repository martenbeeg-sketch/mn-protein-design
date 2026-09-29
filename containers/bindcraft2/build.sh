#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINER_BUILDER="$APP_ROOT/../mn-tool-containers/scripts/build-bindcraft2.sh"

if [[ ! -x "$CONTAINER_BUILDER" ]]; then
  printf 'Clone mn-tool-containers beside mn-protein-design first.\n' >&2
  exit 1
fi

exec "$CONTAINER_BUILDER" "$@"
