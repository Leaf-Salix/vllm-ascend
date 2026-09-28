#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
source "$workspace/env-dsv4-0251rc1.sh"
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_score_key_l1_20260928"
task=task_20260928_175014_22857933418
resume=task_20260928_180158_266927127827
# Keep the original device-open failure explicit; only the missing short DFX
# came from the second task. State and formal timing remain from the first.
[[ "$(task-submit --status "$task")" == 'completed (exit=1)' ]]
[[ "$(task-submit --status "$resume")" == 'completed (exit=0)' ]]
for history in 131072 8192; do
    python "$repo/tests/pypto_test/results/csa_short_score_sync_20260928/summarize.py" \
        --root "$root" --output "$root/h${history}_b16/summary.json" \
        --history "$history" --batch 16 --iters 20 --windows 4 \
        --task "$task; missing 8K DFX only: $resume" \
        --operator '2d2f9ca0 + long S6 dedicated Key L1 with L0B prefetch; other four Score groups unchanged'
done
python "$repo/tests/pypto_test/results/csa_score_key_prefetch_20260928/evidence.py" --root "$root"
