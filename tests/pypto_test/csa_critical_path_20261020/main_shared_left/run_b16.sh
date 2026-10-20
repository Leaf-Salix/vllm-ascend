#!/usr/bin/env bash
# 主压缩两路共享Left，冻结v2源和工具链，完整三方五窗。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment=$(cd -- "$here/.." && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?通过task-submit --device auto运行}"
exec bash "$experiment/run_isolated.sh" "$experiment/run_full.py" \
  --output "$here/device_b16_v2" --batch 16 --history 131072 \
  --launches 3 --l1-windows 5 --wait-backoff --failure-snapshot --metadata-replay --padding-replay \
  --spmd-source "$here/source_candidate_v2" \
  --source-matrix-reference "$experiment/source_after_score_split" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
