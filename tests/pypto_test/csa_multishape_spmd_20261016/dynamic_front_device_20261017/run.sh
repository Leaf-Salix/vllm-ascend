#!/usr/bin/env bash
# CPU 准备和设备执行共用隔离环境；设备模式必须由队列给出 TASK_DEVICE。
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
mode="${1:?指定 prepare 或 execute}"
out="${2:?指定新的结果目录}"
if [[ -e "$out" ]]; then
    printf '拒绝覆盖已有结果：%s\n' "$out" >&2
    exit 1
fi
case "$mode" in
    prepare) extra=(--compile-reference) ;;
    execute) : "${TASK_DEVICE:?只能通过 task-submit --device auto 分配设备}"; extra=(--execute) ;;
    *) printf '不支持的模式：%s\n' "$mode" >&2; exit 1 ;;
esac
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export CSA_STATIC_PYPTO_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
export HCA_SIMPLER_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=1 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
python "$repo/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$here/run_device.py" --output "$out" "${extra[@]}"
