#!/usr/bin/env bash
# 单卡采同形状的正式第二个 C4 层权重，合成输入/历史；不作整模型性能计时。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配一张卡}"
history="${1:?指定 131072 或 8192}"
batch="${2:?指定对应 batch}"
case "$history/$batch" in 131072/4|131072/8|131072/16|8192/24|8192/32|8192/40) ;; *) exit 2 ;; esac
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
case_dir="$repo_root/tests/pypto_test/results/csa_baseline_20260926/six_case_swimlanes/h${history}_b${batch}"
if [[ -d "$case_dir" ]]; then
    printf '已有采集目录，请保留结果并使用新的目录：%s\n' "$case_dir" >&2
    exit 2
fi
mkdir -p "$case_dir"
source "$repo_root/../env-dsv4-0251rc1.sh"
export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
export DYNAMIC_EPLB=false
export EXPERT_MAP_RECORD=false
cd "$case_dir"
python "$repo_root/tests/pypto_test/dsv4_csa_single_layer.py" \
    --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
    --output "$case_dir" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
    --layer-index 4 --variant performance --weight-nz-mode 2 --atomic-add 1 \
    --deterministic-level 0 --swimlane > "$case_dir/run.log" 2>&1
