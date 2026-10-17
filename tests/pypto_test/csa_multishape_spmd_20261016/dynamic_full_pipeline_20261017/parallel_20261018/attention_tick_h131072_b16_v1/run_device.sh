#!/usr/bin/env bash
# 本地隔离诊断；只允许队列分配设备，不安装或改写共享依赖。
set -eo pipefail
csa_evidence=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
csa_workspace=/data/pyptouser/qinchuanyu/pto-eager
csa_output=${1:?指定全新绝对输出目录}
shift
: "${TASK_DEVICE:?只允许 task-submit --device auto 分配设备}"
if [[ -e "$csa_output" || "$csa_output" != /* ]]; then
    printf '输出必须是全新绝对路径：%s\n' "$csa_output" >&2
    exit 1
fi
source "$csa_workspace/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$csa_workspace/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$csa_workspace/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export PYPTO_PROG_BUILD_DIR="$csa_output/build_output" PYPTO_CACHE_DIR="$csa_output/cache"
unset HC_PRE_TEST_OPERATOR_SOURCE
python "$csa_workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$csa_evidence/run_tick_diagnostic.py" --output "$csa_output" "$@"
