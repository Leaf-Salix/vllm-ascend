#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit 分配单卡}"
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source "$repo_root/../env-dsv4-0251rc1.sh"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
case_root="$repo_root/tests/pypto_test/results/csa_baseline_20260926/precision_review"
for batch in 1 3 4; do
    output="$case_root/uniform_b$batch"
    mkdir -p "$output"
    cd "$output"
    python "$repo_root/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" --synthetic-batch "$batch" --output "$output" --variant performance > "$output/run.log" 2>&1
done
cd "$repo_root"
output="$case_root/b1_full_tail"
mkdir -p "$output"
bash tests/pypto_test/run_csa_single_layer.sh "$output" --batch 1 --history 255 --weight-nz-mode 2 --variant performance --atomic-add 0 --deterministic-level 1 --save-state --graph > "$output/run.log" 2>&1
