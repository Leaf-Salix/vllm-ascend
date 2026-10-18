#!/usr/bin/env bash
# 完整符号编译，不占NPU；只使用既有独立依赖，不修改共享安装。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
out=${1:?指定全新输出目录}
shift
cd -- "$repo_dir"
source "$workspace_dir/env-dsv4-0251rc1.sh"
source "$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018/.claude/skills/testing/load-env.sh"
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
export CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 VLLM_ASCEND_ENABLE_NZ=2
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
  "$here/../../check_cpu.py" --output "$out" \
  --spmd-source "$here/source_candidate" \
  --strategy long --controls-json "$here/controls.json" "$@"
