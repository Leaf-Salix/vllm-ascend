#!/usr/bin/env bash
# 当前正式双QK数学加Key多ND供数，三方统一新编译器。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment=$(cd -- "$here/.." && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?通过task-submit --device auto运行}"
exec bash "$here/run_isolated.sh" "$experiment/run_full.py" \
  --output "$here/device_b28_v1" --batch 28 --history 131072 \
  --launches 3 --l1-windows 5 --wait-backoff --failure-snapshot --metadata-replay --padding-replay \
  --spmd-source "$here/source_candidate" \
  --source-matrix-reference "$experiment/source_after_score_split" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
