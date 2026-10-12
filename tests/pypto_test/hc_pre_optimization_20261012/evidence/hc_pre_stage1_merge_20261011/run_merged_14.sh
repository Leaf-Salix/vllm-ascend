#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过队列执行}"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
port_out=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_stage1_merge_20261011/run_merged_14
mkdir -p "$port_out/ascend"
export ASCEND_PROCESS_LOG_PATH="$port_out/ascend"
cd "$port_out"
exec python /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/dsv4_hc_pre_vector_bench.py \
 --output "$port_out" \
 --kernel-file /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/hc_pre.py \
 --pypto-core /data/pyptouser/qinchuanyu/pto-eager/pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so \
 --reference /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_tail_vectors_20261011/pto_only_baseline_14 \
 --batch-size 2 4 8 16 20 24 28 --seqlen 5 6
