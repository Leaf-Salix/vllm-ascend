#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
out="$repo/tests/pypto_test/results/csa_cache_direct_read_20260927/padding"
mkdir -p "$out"
python "$repo/tests/pypto_test/dsv4_csa_single_layer.py" \
  --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 --output "$out" \
  --device "$TASK_DEVICE" --batch 4 --history 4095 --layer-index 4 \
  --variant performance --weight-nz-mode 2 --atomic-add 0 --deterministic-level 1 \
  --padding-graph > "$out/run.log" 2>&1
