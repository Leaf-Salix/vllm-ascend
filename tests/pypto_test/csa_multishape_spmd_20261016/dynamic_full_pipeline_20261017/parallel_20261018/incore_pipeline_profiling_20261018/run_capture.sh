#!/usr/bin/env bash
# 只采首个完整CSA调用；真实占卡必须经队列，输入与生成物留在本地build_output。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
out=${1:?指定全新输出目录}
shift
: "${TASK_DEVICE:?必须经task-submit分配NPU}"
source "$workspace_dir/env-dsv4-0251rc1.sh"
cd -- "$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
source .claude/skills/testing/load-env.sh
cd -- "$repo_dir"
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
export CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
  "$here/capture_workload.py" --output "$out" --batch 16 --history 131072 --launches 1 \
  --l1-windows 0 --spmd-source "$repo_dir/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd" \
  --wait-backoff "$@"
