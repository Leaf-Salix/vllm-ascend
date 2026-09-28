#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_sparse_sync_20260928
for history in 8192 131072; do
    python "$root/../csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --history "$history" --batch 16 \
        --output "$root/h${history}_b16/summary.json" \
        --task task_20260928_124129_133793627580 \
        --operator 'e58ddc94 + qk_pv sync_start=True; Score/early_resolve unchanged'
done
