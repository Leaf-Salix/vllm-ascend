#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
version="${1:?v4 or v7}"
history="${2:?history}"
batch="${3:?batch}"
kind="${4:-timing}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
case "$version" in
    v4) source_repo="$workspace/.cache/csa-native-v4-da6f474a" ;;
    v7) source_repo="$workspace/.cache/csa-native-v7-9516acbe" ;;
    *) exit 2 ;;
esac
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
case_dir="$repo/tests/pypto_test/results/csa_native_cube_matrix_20260927/$version/h${history}_b${batch}/$kind"
mkdir -p "$case_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
case "$kind" in
    timing)
        extra=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse)
        if [[ "$version" == v7 ]]; then extra+=(--profile); fi
        ;;
    swimlane) extra=(--swimlane --swimlane-graph --swimlane-windows 4) ;;
    *) exit 2 ;;
esac
git -C "$source_repo" rev-parse HEAD > "$case_dir/source_revision.txt"
git -C "$workspace/pypto" rev-parse HEAD > "$case_dir/pypto_revision.txt"
cd "$case_dir"
exec python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
    --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
    --output "$case_dir" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
    --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
    --deterministic-level 0 "${extra[@]}" > "$case_dir/run.log" 2>&1
