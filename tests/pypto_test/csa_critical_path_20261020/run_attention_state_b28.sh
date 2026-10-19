#!/usr/bin/env bash
# 独立64KiB KV与四半段在线状态：原三方完整CSA验证，只经队列运行。
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$(cd -- "$repo/.." && pwd)
: "${TASK_DEVICE:?必须通过task-submit --device auto运行}"
exec bash "$here/run_isolated.sh" "$here/run_full.py" \
  --output "$here/attention_state_h131072_b28_v1" --batch 28 --history 131072 \
  --launches 3 --l1-windows 5 --wait-backoff --metadata-replay --padding-replay \
  --spmd-source "$here/source_attention_state_halves" \
  --source-matrix-reference "$here/source_reference" \
  --candidate-ptoas-root "$workspace/.worktrees/ptoas-csa-repro-20261019/build" "$@"
