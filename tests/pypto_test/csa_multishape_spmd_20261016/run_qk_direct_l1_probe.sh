#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?必须通过task-submit --device auto运行}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root=/data/pyptouser/qinchuanyu/pto-eager
out="${1:?新的结果目录}"
if [[ -e "$out" ]]; then
    printf '拒绝覆盖已有结果：%s\n' "$out" >&2
    exit 1
fi
source "$root/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$root/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$root/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p "$out/ascend"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
cp "$here/qk_direct_l1_probe.py" "$here/qk_direct_l1_probe_entry.py" "$out/"
cd "$out"
python "$here/../csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$out/qk_direct_l1_probe.py" --device "$TASK_DEVICE" --output "$out" > "$out/run.log" 2>&1
