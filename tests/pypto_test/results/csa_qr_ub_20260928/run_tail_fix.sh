#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
root="$repo/tests/pypto_test/results/csa_qr_ub_20260928"
out="$root/tail_fix/tail_b10/candidate"
mkdir -p "$out/ascend"
ln -s "$root/tail_b10/baseline" "$root/tail_fix/tail_b10/baseline"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
cd "$out"
python "$repo/tests/pypto_test/results/csa_cache_accuracy_20260927/accuracy_case.py" \
    "$workspace/.cache/csa-qr-ub-tail-a66255ea" \
    --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
    --output "$out" --device "$TASK_DEVICE" --batch 10 --history 8192 \
    --layer-index 4 --variant performance --weight-nz-mode 2 --seed 1024 \
    --atomic-add 0 --deterministic-level 1 --save-state --graph > "$out/run.log" 2>&1
python "$repo/tests/pypto_test/results/csa_indexer_six_20260928/compare_accuracy.py" \
    --root "$root/tail_fix/tail_b10" \
    --scope 'QR UB residency plus direct UB tail publication; reuse completed deterministic T60 baseline; no timing claim'
