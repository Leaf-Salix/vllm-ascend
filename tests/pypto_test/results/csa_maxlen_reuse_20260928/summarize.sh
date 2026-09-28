#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_maxlen_reuse_20260928"
task=$(cat "$root/layer_task.txt")
[[ "$(task-submit --status "$task")" == 'completed (exit=0)' ]]
source "$workspace/env-dsv4-0251rc1.sh"
for history in 131072 8192; do
    python "$repo/tests/pypto_test/results/csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --output "$root/h${history}_b16/summary.json" \
        --history "$history" --batch 16 --iters 20 --windows 4 --task "$task" \
        --operator 'e33d842a/CANN9.2 baseline; candidate reuses current orchestration maximum length'
done
python "$repo/tests/pypto_test/results/csa_score_key_prefetch_20260928/evidence.py" \
    --root "$root" --long-batch 16 --short-batch 16 --baseline 'e33d842a/CANN9.2'
python "$repo/tests/pypto_test/results/csa_score_balanced_leaves_20260928/incore.py" \
    --root "$root" --title 'CANN9.2复用最大cache长度'
