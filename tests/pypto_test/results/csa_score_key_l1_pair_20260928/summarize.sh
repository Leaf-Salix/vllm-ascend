#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
root="$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928"
task=$(cat "$root/layer_task.txt")
[[ "$(task-submit --status "$task")" == 'completed (exit=0)' ]]
for case_spec in 131072:4 8192:16; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    python "$repo/tests/pypto_test/results/csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --output "$root/h${history}_b${batch}/summary.json" \
        --history "$history" --batch "$batch" --iters 20 --windows 4 --task "$task" \
        --operator '554b3bca + long pair M128/N128 dedicated Key L1 with L0B prefetch; other groups unchanged'
done
python "$repo/tests/pypto_test/results/csa_score_key_prefetch_20260928/evidence.py" \
    --root "$root" --long-batch 4 --short-batch 16 --baseline 554b3bca
