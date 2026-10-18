#!/usr/bin/env bash
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=/data/pyptouser/qinchuanyu/pto-eager
cd "$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
source "$workspace/.worktrees/pypto-csa-static-fixes-20261016/.claude/skills/testing/load-env.sh"
export PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$workspace/.worktrees/simpler-csa-core-args-256-20261016/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
python "$here/run_repro.py" "$@"
