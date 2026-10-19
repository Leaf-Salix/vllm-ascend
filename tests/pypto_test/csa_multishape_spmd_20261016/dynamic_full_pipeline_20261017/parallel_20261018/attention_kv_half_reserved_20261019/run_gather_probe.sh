#!/usr/bin/env bash
# 原多task、正式单SPMD与Attention显式行DMA候选三方对照；必须由队列分配NPU。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
out=${1:?指定全新输出目录}
shift
: "${TASK_DEVICE:?必须经task-submit --device auto分配设备}"
cd -- "$repo_dir"
source "$workspace_dir/env-dsv4-0251rc1.sh"
# 仅本候选进程使用独立PTOAS构建，原共享安装不变。
export PTOAS_ROOT="$here/../score_key_l1_region_sync_20261019/toolchain_v3"
export PATH="$PTOAS_ROOT/bin:$PATH"
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
export CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
unset HC_PRE_TEST_OPERATOR_SOURCE
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
  "$here/gather_probe.py" "$out" "$@"
