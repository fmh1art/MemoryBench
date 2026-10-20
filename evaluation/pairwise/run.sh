#!/usr/bin/env bash
set -euo pipefail
pairwise_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$pairwise_root"
exec "${PAIRWISE_PYTHON:-python}" -m evaluation.pairwise.server "$@"
