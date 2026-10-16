#!/usr/bin/env bash
# 小型数据交接验证，只通过统一队列占卡。
set -eo pipefail
: "${TASK_DEVICE:?必须由 task-submit 分配设备}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root=/data/pyptouser/qinchuanyu/pto-eager
out="${1:?指定全新输出目录}"
source "$root/env-dsv4-0251rc1.sh"
if [[ -e "$out" ]]; then
    printf '拒绝覆盖已有目录：%s\n' "$out" >&2
    exit 1
fi
mkdir -p "$out/ascend"
export CSA_STATIC_PYPTO_ROOT="$root/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$root/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_PROCESS_LOG_PATH="$out/ascend"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
cd "$out"
python "$here/bootstrap_isolated.py" "$here/test_local_signal_payload.py" --output "$out" > "$out/run.log" 2>&1
