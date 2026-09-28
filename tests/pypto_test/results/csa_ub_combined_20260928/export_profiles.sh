#!/usr/bin/env bash
# 只做离线CPU解析；必须等模型任务成功结束，不与正式计时竞争CPU。
set -eo pipefail
model_task=task_20260928_162727_150252329060
status=$(task-submit --status "$model_task")
if [[ "$status" != 'completed (exit=0)' ]]; then
    echo "$model_task: $status; defer profile export" >&2
    exit 1
fi
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-ub-combined-2d2f9ca0"
source "$workspace/env-dsv4-0251rc1.sh"
cd "$repo"
root=tests/pypto_test/results/csa_ub_combined_20260928
for history in 131072 8192; do
    for backend in native pto; do
        python "$source_repo/tests/pypto_test/offline_pd/run.py" profile-export \
            --bank "tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank" \
            --output "$root/model/h$history/b16/$backend" --profile-ranks 0 --analyse-processes 4 \
            > "$root/model/h$history/b16/export_$backend.log" 2>&1
    done
done
python tests/pypto_test/results/csa_source_split_ab_20260927/ordered/analyze_profile.py \
    --root "$root/model" --per-batch --cases 131072:16 8192:16
python "$root/profile_report.py"
