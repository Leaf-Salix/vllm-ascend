#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-qa-adaptive-88d0744f"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_qa_matrix_20260928/swimlanes"
for shape in 131072:4 131072:8 131072:16 8192:16 8192:24 8192:32 8192:40; do
    history="${shape%:*}"
    batch="${shape#*:}"
    out="$root/h${history}_b${batch}"
    mkdir -p "$out/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend"
    cd "$out"
    python "$source_repo/tests/pypto_test/dsv4_csa_single_layer.py" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
        --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 0 --swimlane --swimlane-graph \
        > "$out/run.log" 2>&1
done
