#!/usr/bin/env bash
# 在新 worktree 自己的 venv 安装 binding，再只构建本轮需要的 A3 两套 runtime。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
source "$workspace/env-dsv4-0251rc1.sh"
cd -- "$workspace/.worktrees/simpler-csa-sdma-refresh-20261020"
python -m venv --system-site-packages .venv
source .venv/bin/activate
export CMAKE_BUILD_PARALLEL_LEVEL=2
export PYTHONPATH="$workspace/.venv-dsv4-0251rc1/lib/python3.10/site-packages:${PYTHONPATH:-}"
python -m pip install --no-build-isolation --no-deps -e . \
    --config-settings=build.targets=_task_interface
export CSA_SDMA_SIMPLER_ROOT="$PWD"
unset CSA_SDMA_PYPTO_ROOT
export PYTHONPATH="$repo/tests/pypto_test/csa_sdma_prefetch_20261019/python_startup:$PYTHONPATH"
python -m simpler_setup.build_runtimes --platforms a2a3
