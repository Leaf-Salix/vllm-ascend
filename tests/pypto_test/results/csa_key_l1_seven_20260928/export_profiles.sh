#!/usr/bin/env bash
# 本线程模型及后续B8单卡结束后再离线解析，避免竞争正式计时的CPU。
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-key-l1-seven-554b3bca"
root="$repo/tests/pypto_test/results/csa_key_l1_seven_20260928"
model_task=$(cat "$root/model_task.txt")
status=$(task-submit --status "$model_task")
if [[ "$status" != 'completed (exit=0)' ]]; then
    echo "$model_task: $status; defer profile export" >&2
    exit 1
fi
candidate_task=$(cat "$root/../csa_score_key_l1_only_20260928/layer_task.txt")
candidate_status=$(task-submit --status "$candidate_task")
if [[ "$candidate_status" != completed* ]]; then
    echo "$candidate_task: $candidate_status; defer CPU export until the queued single-card run ends" >&2
    exit 1
fi
source "$workspace/env-dsv4-0251rc1.sh"
cd "$repo"
cases=(131072:4 131072:8 131072:16 8192:16 8192:24 8192:32 8192:40)
for case_spec in "${cases[@]}"; do
    history="${case_spec%:*}"
    batch="${case_spec#*:}"
    for backend in native pto; do
        python "$source_repo/tests/pypto_test/offline_pd/run.py" profile-export \
            --bank "tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank" \
            --output "$root/model/h$history/b$batch/$backend" --profile-ranks 0 --analyse-processes 4 \
            > "$root/model/h$history/b$batch/export_$backend.log" 2>&1
    done
done
python tests/pypto_test/results/csa_source_split_ab_20260927/ordered/analyze_profile.py \
    --root "$root/model" --per-batch --cases "${cases[@]}"
python "$root/profile_report.py"
