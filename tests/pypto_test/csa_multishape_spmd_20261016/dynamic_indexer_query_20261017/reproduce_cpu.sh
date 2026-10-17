#!/usr/bin/env bash
# 只编译动态数学helper，不执行NPU；输出目录不得已存在。
set -eo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
out="${1:?指定新的CPU结果目录}"
shift
if [[ -e "$out" ]]; then
    printf '拒绝覆盖：%s\n' "$out" >&2
    exit 1
fi
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export CSA_STATIC_PYPTO_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
export HCA_SIMPLER_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python "$repo/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$here/check_cpu.py" --output "$out" "$@"
