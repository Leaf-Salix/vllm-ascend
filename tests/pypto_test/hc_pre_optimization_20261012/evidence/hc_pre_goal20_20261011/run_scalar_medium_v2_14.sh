#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?仅经队列运行}"
cd /data/pyptouser/qinchuanyu/pto-eager
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
candidate_name=mixed_scalar_medium_v2
candidate_batch=all
candidate_seqlen=all
candidate_label=scalar_medium_v2_14
candidate_output="$candidate_root/$candidate_label"
export PYPTO_PROG_BUILD_DIR="$candidate_root/jit_$candidate_label"
export PYPTO_CACHE_DIR="$candidate_root/cache_$candidate_label"
mkdir -p "$candidate_output/ascend"
export ASCEND_PROCESS_LOG_PATH="$candidate_output/ascend"
exec python /data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-stage1-db-20261011/tests/pypto_test/dsv4_hc_pre_bench.py \
 --kernel-file "$candidate_root/$candidate_name.py" --output "$candidate_output" \
 --pypto-core /data/pyptouser/qinchuanyu/pto-eager/pypto/build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so \
 --native-library /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/torch-hc-pre-build/hc_pre_recipes_reference.so \
 --native-opp /data/pyptouser/qinchuanyu/pto-eager/.cache/hc-pre-recipes-20261009/ops/ascendc/build/_CPack_Packages/Linux/External/CANN-custom_ops-none-linux.aarch64.run/packages/vendors/customize \
 --timing incore --swimlane-level 1 --warmup 5 --batch-size 2 4 8 16 20 24 28 --seqlen 5 6 \
 > "$candidate_root/$candidate_label.log" 2>&1
