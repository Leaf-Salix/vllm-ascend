#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
root="$repo/tests/pypto_test/results/csa_score_balanced_leaves_20260928"
task=$(cat "$root/layer_task.txt")
[[ "$(task-submit --status "$task")" == 'completed (exit=0)' ]]
for case_spec in 131072:16 8192:16; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    python "$repo/tests/pypto_test/results/csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --output "$root/h${history}_b${batch}/summary.json" \
        --history "$history" --batch "$batch" --iters 20 --windows 4 --task "$task" \
        --operator '8e176285 + balanced long leaves for 16 query groups; no pair arena growth; short Score unchanged'
done
python "$repo/tests/pypto_test/results/csa_score_key_prefetch_20260928/evidence.py" \
    --root "$root" --long-batch 16 --short-batch 16 --baseline 8e176285
