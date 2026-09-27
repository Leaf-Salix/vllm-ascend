#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-cache-pressure-f76b3ad4"
out="$repo/tests/pypto_test/results/csa_model_forward_f76b3ad4_20260927/cache_pressure"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
mkdir -p "$out/case/ascend"
export ASCEND_PROCESS_LOG_PATH="$out/case/ascend"
cd "$out/case"
exec python "$out/diagnose.py" "$source_repo" \
    --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 --output "$out/case" \
    --device "$TASK_DEVICE" --batch 16 --history 131072 --layer-index 4 \
    --variant performance --weight-nz-mode 2 --atomic-add 1 --deterministic-level 0 \
    --timing-iters 20 --timing-warmup 5 --timing-metadata reuse > "$out/run.log" 2>&1
