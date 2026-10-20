#!/usr/bin/env bash
# 当前数学的动态wave候选：功能先过冻结门槛，再采同轮完整三方五窗。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment=$(cd -- "$here/.." && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?必须通过task-submit --device auto运行}"
exec bash "$experiment/run_isolated.sh" "$experiment/run_full.py" \
  --output "$here/device_b16_v1" --batch 16 --history 131072 \
  --launches 3 --l1-windows 5 --wait-backoff --metadata-replay --padding-replay \
  --spmd-source "$here/source_candidate" \
  --source-matrix-reference "$experiment/source_after_score_split" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
