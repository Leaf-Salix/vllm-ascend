#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过任务队列执行}"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
clock_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results
clock_out="$clock_root/hc_pre_tail_perf_20261011/run_clock_probe_parts_forced"
mkdir -p "$clock_out/ascend"
export ASCEND_PROCESS_LOG_PATH="$clock_out/ascend"
cd "$clock_out"
exec python "$clock_root/hc_pre_tail_perf_20261011/clock_diagnostic_launcher.py" run --output "$clock_out" --kernel-file "$clock_root/hc_pre_tail_perf_20261011/probe_tail_parts_forced.py" --pypto-core /data/pyptouser/qinchuanyu/pto-eager/pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so --batch-size 2 8 16 28 --seqlen 6 --reference "$clock_root/hc_pre_tail_vectors_20261011/pto_only_baseline_14" --probe --expected-active-vectors 48
