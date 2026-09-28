#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
source "$workspace/env-dsv4-0251rc1.sh"
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_score_key_prefetch_20260928"
task=task_20260928_170324_293095832121
[[ "$(task-submit --status "$task")" == 'completed (exit=0)' ]]
for history in 131072 8192; do
    python "$repo/tests/pypto_test/results/csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --output "$root/h${history}_b16/summary.json" \
        --history "$history" --batch 16 --iters 20 --windows 4 --task "$task" \
        --operator '2d2f9ca0 + long S6 Key L0B prefetch; other four Score groups unchanged'
done
python "$root/evidence.py"
