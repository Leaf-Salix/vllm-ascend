#!/usr/bin/env bash
# 完整CSA三方五窗；只经队列运行。算子、runner和工具链在任务期间冻结。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?必须通过task-submit --device auto运行}"
out="$here/qb_stream_h131072_b28_v3"
exec bash "$here/run_isolated.sh" "$here/run_full.py" \
  --output "$out" --batch 28 --history 131072 --launches 3 --l1-windows 5 \
  --wait-backoff --metadata-replay --padding-replay \
  --spmd-source "$here/source_qb_stream_reuse" \
  --source-matrix-reference "$here/source_reference" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
