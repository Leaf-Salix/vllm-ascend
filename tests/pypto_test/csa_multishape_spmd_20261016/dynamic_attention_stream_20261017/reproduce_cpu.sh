#!/usr/bin/env bash
# 全部工作仅在CPU侧编译；参数为新结果目录名，禁止覆盖已有尝试。
set -euo pipefail
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$experiment_dir/../../../.." && pwd)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
output_name=${1:?请指定新的产物目录名}
set +u
source "$workspace_dir/env-dsv4-0251rc1.sh"
set -u
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_ASCEND_ENABLE_NZ=2
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
  "$experiment_dir/check_cpu.py" --output "$experiment_dir/$output_name"
