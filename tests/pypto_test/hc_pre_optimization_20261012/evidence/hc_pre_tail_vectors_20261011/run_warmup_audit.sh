#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过 task-submit --device auto 提交}"
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
output_dir="/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_tail_vectors_20261011/baseline_b16_warmup_audit"
mkdir -p "$output_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$output_dir/ascend"
cd "$output_dir"
exec python /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_tail_vectors_20261011/audit_warmup.py --output "$output_dir" --kernel-file /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_tail_vectors_20261011/baseline.py --batch-size 16 --seqlen 6 --pypto-core /data/pyptouser/qinchuanyu/pto-eager/pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so --native-library /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/torch-hc-pre-build/hc_pre_recipes_reference.so --native-opp /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/ops/ascendc/build/_CPack_Packages/Linux/External/CANN-custom_ops-none-linux.aarch64.run/packages/vendors/customize
