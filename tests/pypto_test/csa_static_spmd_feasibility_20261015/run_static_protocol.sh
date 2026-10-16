#!/usr/bin/env bash
# 只诊断既有 128K/B16 的跨核控制表；无 CSA 数学或性能结论。
set -eo pipefail
: "${TASK_DEVICE:?必须经 task-submit --device auto}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:?独立诊断输出目录}"
root=/data/pyptouser/qinchuanyu/pto-eager
source "$root/env-dsv4-0251rc1.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CSA_STATIC_PYPTO_ROOT="$root/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$root/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
mkdir -p "$out/ascend"
cd "$out"
python "$here/bootstrap_isolated.py" "$here/diagnose_static_protocol.py" \
    --case "$here/static/h131072_b16" --output "$out" --compiled "$out/cpu/kernel" > "$out/run.log" 2>&1
