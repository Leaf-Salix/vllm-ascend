#!/usr/bin/env bash
# 仅使用独立 Python overlay；不触发设备执行或改写已安装包。
set -eo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dependency_root=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
source "$dependency_root/.claude/skills/testing/load-env.sh"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$task_root/pypto_overlay/python:/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
python "$task_root/reproduce_tile.py" "$@"
