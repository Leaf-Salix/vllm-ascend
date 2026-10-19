#!/usr/bin/env bash
# 对同一正式候选检查可选 GM 子视图扩展关闭时的完整代码生成；不占 NPU。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
out=${1:?指定全新输出目录}
shift
source "$workspace_dir/env-dsv4-0251rc1.sh"
cd -- "$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
source .claude/skills/testing/load-env.sh
cd -- "$repo_dir"
export PTOAS_ROOT="$here/../score_key_l1_region_sync_20261019/toolchain_v3"
export PATH="$PTOAS_ROOT/bin:$PATH"
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
export CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 VLLM_ASCEND_ENABLE_NZ=2
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
  "$here/flatten_gm_view_repro.py" "$out" "$@"
