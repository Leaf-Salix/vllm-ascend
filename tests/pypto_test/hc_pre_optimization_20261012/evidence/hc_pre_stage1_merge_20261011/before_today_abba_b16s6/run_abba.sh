#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?通过队列执行}"
cd /data/pyptouser/qinchuanyu/pto-eager
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
compare_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_stage1_merge_20261011/before_today_abba_b16s6
for compare_pass in A1 B1 B2 A2
do
 case "$compare_pass" in
  A*) kernel_source="$compare_root/A_before_today.py" ;;
  B*) kernel_source="$compare_root/B_current.py" ;;
 esac
 export PYPTO_PROG_BUILD_DIR="$compare_root/jit_$compare_pass"
 export PYPTO_CACHE_DIR="$compare_root/cache_$compare_pass"
 mkdir -p "$compare_root/$compare_pass/ascend"
 export ASCEND_PROCESS_LOG_PATH="$compare_root/$compare_pass/ascend"
 python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-stage1-db-20261011/tests/pypto_test/dsv4_hc_pre_bench.py \
  --kernel-file "$kernel_source" --output "$compare_root/$compare_pass" \
  --pypto-core /data/pyptouser/qinchuanyu/pto-eager/pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so \
  --native-library /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/torch-hc-pre-build/hc_pre_recipes_reference.so \
  --native-opp /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/ops/ascendc/build/_CPack_Packages/Linux/External/CANN-custom_ops-none-linux.aarch64.run/packages/vendors/customize \
  --timing incore --swimlane-level 1 --batch-size 16 --seqlen 6 \
  > "$compare_root/$compare_pass.log" 2>&1
done
