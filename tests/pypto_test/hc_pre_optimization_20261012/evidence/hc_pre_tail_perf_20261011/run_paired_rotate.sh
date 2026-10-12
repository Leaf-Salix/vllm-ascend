#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过队列执行}"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
pair_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_tail_perf_20261011
pair_out="$pair_root/run_paired_rotate_medium_14"
mkdir -p "$pair_out/ascend"
export ASCEND_PROCESS_LOG_PATH="$pair_out/ascend"
cd "$pair_out"
exec python "$pair_root/paired_l1_bench.py" --output "$pair_out" --baseline "$pair_root/baseline48.py" --candidate "$pair_root/mix_rotate_medium_v2.py"
