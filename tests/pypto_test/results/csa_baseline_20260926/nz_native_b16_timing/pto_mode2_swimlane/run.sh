#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
: "${TASK_DEVICE:?须经 task-submit 提交}"
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=1
export PTO_CSA_VARIANT=performance
export HCCL_DETERMINISTIC=false
repo_root="$PTO_EAGER_ROOT/vllm-ascend-dsv4-pto-0251rc1"
case_root="$repo_root/tests/pypto_test/results/csa_baseline_20260926/nz_native_b16_timing"
cd "$case_root/pto_mode2_swimlane"
export ASCEND_PROCESS_LOG_PATH="$PWD/ascend"
python "$repo_root/tests/pypto_test/dsv4_csa_single_card_bench.py" \
    --args-dir "$case_root/mode2/case" --output "$PWD" --device 0 \
    --warmup 1 --iters 1 --windows 1 --swimlane 4 > run.log 2>&1
