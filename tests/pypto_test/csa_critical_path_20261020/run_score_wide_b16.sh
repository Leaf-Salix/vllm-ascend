#!/usr/bin/env bash
# 私有Cube实际N128宽面板候选；只经队列运行三方完整CSA五窗。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?必须通过task-submit --device auto运行}"
exec bash "$here/run_isolated.sh" "$here/run_full.py" \
  --output "$here/score_wide_h131072_b16_v1" --batch 16 --history 131072 \
  --launches 3 --l1-windows 5 --wait-backoff --metadata-replay --padding-replay \
  --spmd-source "$here/source_score_wide_pipeline" \
  --source-matrix-reference "$here/source_after_score_split" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
