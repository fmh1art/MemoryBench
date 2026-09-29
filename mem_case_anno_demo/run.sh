#!/usr/bin/env bash
set -euo pipefail
ANNOTATOR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ANNOTATOR_PYTHON="${ACWM_ANNOTATOR_PYTHON:-python}"
exec "$ANNOTATOR_PYTHON" "$ANNOTATOR_DIR/server.py" "$@"
