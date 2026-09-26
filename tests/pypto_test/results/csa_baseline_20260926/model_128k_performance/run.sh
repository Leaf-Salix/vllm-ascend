#!/usr/bin/env bash
# 泛化矩阵分两组历史：每组各侧加载一次；容量固定 40，EPLB 关闭。
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配 16 卡}"
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
# 队列会追加 --device 参数；只消费本脚本的 history，设备由 TASK_DEVICE 传给 launcher。
history=131072
if [[ "${1:-}" != "--device" && -n "${1:-}" ]]; then history="$1"; fi
case "$history" in
    131072) batches=(4 8 16); budget=256; group=model_128k_performance ;;
    8192) batches=(24 32 40); budget=400; group=model_8k_large_batch_performance ;;
    *) exit 2 ;;
esac
result_root="$repo_root/tests/pypto_test/results/csa_baseline_20260926/$group/capacity40"
cd "$repo_root"
source ../env-dsv4-0251rc1.sh
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=1
export HCCL_DETERMINISTIC=false
export DYNAMIC_EPLB=false
export EXPERT_MAP_RECORD=false
export ASCEND_PROCESS_LOG_PATH="$result_root/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
for backend in native pto; do
    python tests/pypto_test/offline_pd/run.py performance \
        --bank "tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank" \
        --output "$result_root/$backend" --backend "$backend" \
        --batch 40 --sweep-batches "${batches[@]}" --max-num-batched-tokens "$budget" \
        --decode-tokens 128 --weight-nz-mode 2 \
        --warmup-rounds 1 --warmup-tokens 96 --warmup-steps 8 --steady-cycles 10 \
        --profile-start-step 8 --profile-steps 3 \
        --graph-mode full_decode_only --capture-sizes 24 48 96 144 192 240
done
