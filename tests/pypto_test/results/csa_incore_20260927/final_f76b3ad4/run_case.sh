#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-cache-f76b3ad4"
history="${1:?history}"
batch="${2:?batch}"
kind="${3:?timing, swimlane, or profile}"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
out="$repo/tests/pypto_test/results/csa_incore_20260927/final_f76b3ad4/h${history}_b${batch}/$kind"
mkdir -p "$out/ascend"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
case "$kind" in
  timing) extra=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse --profile) ;;
  swimlane) extra=(--swimlane --swimlane-graph --swimlane-windows 4) ;;
  profile) extra=(--timing-iters 1 --timing-warmup 5 --timing-metadata reuse --profile) ;;
  *) exit 2 ;;
esac
cd "$out"
exec python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
  --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 --output "$out" \
  --device "$TASK_DEVICE" --batch "$batch" --history "$history" --layer-index 4 \
  --variant performance --weight-nz-mode 2 --atomic-add 1 --deterministic-level 0 \
  "${extra[@]}" > "$out/run.log" 2>&1
