#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_topk_fourway_20260928
for history in 131072 8192; do
    python "$root/../csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --history "$history" --batch 16 --iters 20 \
        --output "$root/h${history}_b16/summary.json" \
        --task task_20260928_125318_172416512553 \
        --operator 'e58ddc94 + QLI V2 four-way Top-K; scheduling unchanged'
done
