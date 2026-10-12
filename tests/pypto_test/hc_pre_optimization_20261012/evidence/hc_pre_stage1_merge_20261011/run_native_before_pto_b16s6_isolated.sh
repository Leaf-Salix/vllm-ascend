#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过队列执行}"
cd /data/pyptouser/qinchuanyu/pto-eager
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
measure_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_stage1_merge_20261011
export PYPTO_PROG_BUILD_DIR="$measure_root/jit_native_before_pto_b16s6_isolated"
export PYPTO_CACHE_DIR="$measure_root/cache_native_before_pto_b16s6_isolated"
mkdir -p "$measure_root/run_native_before_pto_b16s6_isolated/ascend"
export ASCEND_PROCESS_LOG_PATH="$measure_root/run_native_before_pto_b16s6_isolated/ascend"
exec python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-stage1-db-20261011/tests/pypto_test/dsv4_hc_pre_bench.py \
 --kernel-file /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/vllm_ascend/ops/pypto/hc_pre.py \
 --output "$measure_root/run_native_before_pto_b16s6_isolated" \
 --pypto-core /data/pyptouser/qinchuanyu/pto-eager/pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so \
 --native-library /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/torch-hc-pre-build/hc_pre_recipes_reference.so \
 --native-opp /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/ops/ascendc/build/_CPack_Packages/Linux/External/CANN-custom_ops-none-linux.aarch64.run/packages/vendors/customize \
 --timing incore --swimlane-level 1 --batch-size 16 --seqlen 6
