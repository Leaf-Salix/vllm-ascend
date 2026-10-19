#!/usr/bin/env bash
# 新正式Score之上的KV核分配候选，原三方五窗完整CSA门槛。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?必须通过task-submit --device auto运行}"
exec bash "$here/run_isolated.sh" "$here/run_full.py" \
  --output "$here/prefix_kv_h131072_b28_v1" --batch 28 --history 131072 \
  --launches 3 --l1-windows 5 --wait-backoff --metadata-replay --padding-replay \
  --spmd-source "$here/source_prefix_kv_balance" \
  --source-matrix-reference "$here/source_after_score_split" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
