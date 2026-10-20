#!/usr/bin/env bash
# 仅采当前正式与原多task的PMU；原驱动的L1开关只用于进入替换后的回调。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment=$(cd -- "$here/.." && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?通过task-submit --device auto运行}"
exec bash "$experiment/run_isolated.sh" "$here/run_profile.py" \
  --output "$here/device_b16_v1" --batch 16 --history 131072 \
  --launches 1 --l1-windows 3 --wait-backoff --failure-snapshot \
  --spmd-source "$experiment/source_after_score_split" \
  --source-matrix-reference "$experiment/source_after_score_split" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
