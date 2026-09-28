#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_ub_combined_20260928
for history in 131072 8192; do
    python "$root/../csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --history "$history" --batch 16 --iters 20 --windows 2 \
        --output "$root/h${history}_b16/summary.json" \
        --task task_20260928_160301_64273522658 \
        --operator 'd1f170ff vs 2d2f9ca0; Top-K UB + fixed QR UB; long timing overlapped CPU compile, see README'
done
