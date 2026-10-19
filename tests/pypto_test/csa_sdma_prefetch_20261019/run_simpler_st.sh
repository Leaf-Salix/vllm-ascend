#!/usr/bin/env bash
# 队列分配一张 A3；两个 runtime 分别启动独立进程验证原 PR 的 SDMA 用例。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
runtime=${1:?指定 tensormap_and_ringbuffer 或 host_build_graph}
: "${TASK_DEVICE:?必须通过 task-submit --device auto 运行}"
case "$runtime" in
  tensormap_and_ringbuffer|host_build_graph) ;;
  *) exit 2 ;;
esac
source "$workspace/env-dsv4-0251rc1.sh"
toolchain="$workspace/.worktrees/simpler-csa-sdma-20261019"
source "$toolchain/.venv/bin/activate"
export CSA_SDMA_SIMPLER_ROOT="$toolchain"
export PYTHONPATH="$here/python_startup:$toolchain/python:$toolchain:$toolchain/.venv/lib/python3.10/site-packages:$workspace/.venv-dsv4-0251rc1/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export ASCEND_PROCESS_LOG_PATH="$here/simpler_${runtime}/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
cd -- "$toolchain"
python -m pytest "tests/st/a2a3/$runtime/kernel_mode_capture/test_kernel_prefetch.py" \
  --platform a2a3 --runtime "$runtime" --device "$TASK_DEVICE" \
  --pto-session-timeout 900 -q \
  --junitxml="$here/simpler_${runtime}/junit.xml"
