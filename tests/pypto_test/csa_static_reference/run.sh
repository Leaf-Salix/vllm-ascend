#!/usr/bin/env bash
# 独立静态CSA参考的测试驱动，算子源码仅来自ops归档包。
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$here/../../.." && pwd)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
out=${1:?指定全新结果目录}
shift
: "${TASK_DEVICE:?必须经task-submit --device auto运行}"
case "$out" in /*) ;; *) out=$here/$out ;; esac
if [[ -e "$out" ]]; then
    printf '拒绝覆盖已有结果：%s\n' "$out" >&2
    exit 1
fi
cd -- "$repo_dir"
set +u
source "$workspace_dir/env-dsv4-0251rc1.sh"
set -u
export CSA_STATIC_PYPTO_ROOT=${CSA_STATIC_PYPTO_ROOT:-"$workspace_dir/.worktrees/pypto-csa-static-fixes-20261016"}
export CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT=${HCA_SIMPLER_ROOT:-"$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"}
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
unset HC_PRE_TEST_OPERATOR_SOURCE
python "$here/bootstrap_isolated.py" "$here/run.py" --output "$out" "$@"
