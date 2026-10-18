#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$bench_dir/../.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    benchmark_python="$PYTHON"
elif [[ -x "$repo_dir/.venv/bin/python" ]]; then
    benchmark_python="$repo_dir/.venv/bin/python"
else
    benchmark_python=python3
fi
exec "$benchmark_python" "$bench_dir/run_experiment.py" "$@"
