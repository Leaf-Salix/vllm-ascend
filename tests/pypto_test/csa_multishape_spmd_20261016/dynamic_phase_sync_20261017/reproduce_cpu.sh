#!/usr/bin/env bash
# 使用现有独立依赖 worktree，只做 CPU 编译；每次使用新输出名保留失败证据。
set -euo pipefail
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$probe_dir/../../../.." && pwd)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
output_name=${1:?请提供新的输出目录名}
source "$workspace_dir/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$probe_dir/check_cpu.py" --output "$probe_dir/$output_name"
