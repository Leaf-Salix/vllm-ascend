#!/usr/bin/env bash
# 隔离Tile注解overlay的全链CPU入口；只准备入口，需主会话通知后才执行。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base_dir=$(cd -- "$here/../.." && pwd)
repo_dir=$(cd -- "$base_dir/../../../.." && pwd)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
out=${1:?指定全新结果目录}
shift
case "$out" in /*) ;; *) out=$here/$out ;; esac
if [[ -e "$out" ]]; then
    printf '结果目录已经存在：%s\n' "$out" >&2
    exit 1
fi
# 既有环境激活会查询仓库状态；入口允许从workspace等任意cwd调用。
cd -- "$repo_dir"
source "$workspace_dir/env-dsv4-0251rc1.sh"
source "$workspace_dir/.worktrees/pypto-csa-static-fixes-20261016/.claude/skills/testing/load-env.sh"
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/jit_tile_parameter_overlay_20261018/pypto_overlay"
export CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 VLLM_ASCEND_ENABLE_NZ=2
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$here/check_cpu.py" --output "$out" "$@"
