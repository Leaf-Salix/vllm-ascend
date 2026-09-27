#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
out="$repo/tests/pypto_test/results/csa_cache_source_split_20260927"
python "$out/check_native.py" --device "$TASK_DEVICE" \
  --library "$repo/.cache/csa/native-install/vllm_ascend_C.cpython-310-aarch64-linux-gnu.so" \
  --output "$out/native_check.json" > "$out/run.log" 2>&1
