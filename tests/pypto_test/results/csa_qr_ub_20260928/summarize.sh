#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_qr_ub_20260928
for history in 8192 131072; do
    python "$root/../csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --history "$history" --batch 16 --iters 20 \
        --output "$root/h${history}_b16/summary.json" \
        --task task_20260928_143910_419318230381 \
        --operator 'd1f170ff vs a66255ea + QR input/gamma UB reuse; arithmetic/scheduler unchanged'
done
