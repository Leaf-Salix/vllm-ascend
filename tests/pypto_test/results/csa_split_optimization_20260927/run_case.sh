#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
label="${1:?candidate label}"
history="${2:?history}"
batch="${3:?batch}"
kind="${4:-timing}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$repo"
if [[ "$label" == v8_native_pair ]]; then
    source_repo="$workspace/.cache/csa-native-v8-05e0b518"
fi
if [[ "$label" == baseline_cd1fdaa1 ]]; then
    source_repo="$workspace/.cache/csa-split-baseline-cd1fdaa1"
fi
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
case_dir="$repo/tests/pypto_test/results/csa_split_optimization_20260927/$label/h${history}_b${batch}/$kind"
mkdir -p "$case_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
case "$kind" in
    timing) extra=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse) ;;
    swimlane) extra=(--swimlane --swimlane-graph --swimlane-windows 4) ;;
    *) exit 2 ;;
esac
cd "$case_dir"
exec python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
    --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
    --output "$case_dir" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
    --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
    --deterministic-level 0 "${extra[@]}" > "$case_dir/run.log" 2>&1
