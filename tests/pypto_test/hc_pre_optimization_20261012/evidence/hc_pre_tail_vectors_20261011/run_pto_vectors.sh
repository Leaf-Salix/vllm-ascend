#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit --device auto 执行}"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
vector_output="${1:?新的结果目录}"
shift
mkdir -p "$vector_output/ascend"
export ASCEND_PROCESS_LOG_PATH="$vector_output/ascend"
cd "$vector_output"
exec python /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/dsv4_hc_pre_vector_bench.py --output "$vector_output" "$@"
