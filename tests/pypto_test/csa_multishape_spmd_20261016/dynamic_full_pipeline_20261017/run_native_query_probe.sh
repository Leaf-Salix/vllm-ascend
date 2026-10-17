#!/usr/bin/env bash
# Native只做S5正确性参照；不测性能、不改变原S6数学配置或共享安装。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../../.." && pwd)
workspace=$(cd -- "$repo/.." && pwd)
out=${1:?指定全新结果目录}
shift
: "${TASK_DEVICE:?只允许task-submit --device auto分配设备}"
if [[ -e "$out" ]]; then
    printf '结果目录已经存在：%s\n' "$out" >&2
    exit 1
fi
source "$workspace/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$workspace/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$workspace/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
unset HC_PRE_TEST_OPERATOR_SOURCE
python "$repo/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$here/native_query_probe.py" --output "$out" "$@"
