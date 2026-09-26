#!/usr/bin/env bash
# 第二个 CSA 层；单卡完整图区间，先检查长上下文代表档和大 batch 短上下文。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配 1 卡}"
label="${1:?指定本轮标签}"
case "$label" in baseline|score_sync_start) ;; *) exit 2 ;; esac
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source "$repo_root/../env-dsv4-0251rc1.sh"
export DYNAMIC_EPLB=false
export EXPERT_MAP_RECORD=false
for spec in 131072/16 8192/40; do
    history="${spec%/*}"
    batch="${spec#*/}"
    case_dir="$repo_root/tests/pypto_test/results/csa_baseline_20260926/long_context_dispatch/$label/h${history}_b${batch}"
    if [[ -d "$case_dir" ]]; then
        printf '已有结果目录：%s\n' "$case_dir" >&2
        exit 2
    fi
    mkdir -p "$case_dir"
    export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
    cd "$case_dir"
    python "$repo_root/tests/pypto_test/dsv4_csa_single_layer.py" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$case_dir" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
        --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
        --deterministic-level 0 --timing-iters 20 --timing-warmup 5 --timing-metadata reuse \
        > "$case_dir/run.log" 2>&1
done
