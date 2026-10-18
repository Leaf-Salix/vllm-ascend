#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
source "$workspace_dir/env-dsv4-0251rc1.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export CMAKE_BUILD_PARALLEL_LEVEL=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd -- "$repo_dir"
python "$here/run_profile.py"
