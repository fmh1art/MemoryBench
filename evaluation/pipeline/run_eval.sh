#!/usr/bin/env bash
set -euo pipefail
MEM_BENCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MEM_BENCH_PYTHON="${MEM_BENCH_PYTHON:-python}"
exec "$MEM_BENCH_PYTHON" "$MEM_BENCH_DIR/evaluate.py" "$@"
