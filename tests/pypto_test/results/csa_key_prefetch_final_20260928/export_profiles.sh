#!/usr/bin/env bash
# 定向模型结束后离线解析，避免竞争正式计时的CPU。
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-key-prefetch-final-8e176285"
root="$repo/tests/pypto_test/results/csa_key_prefetch_final_20260928"
model_task=$(cat "$root/model_task.txt")
status=$(task-submit --status "$model_task")
if [[ "$status" != 'completed (exit=0)' ]]; then
    echo "$model_task: $status; defer profile export" >&2
    exit 1
fi
source "$workspace/env-dsv4-0251rc1.sh"
cd "$repo"
cases=(131072:4 131072:8 8192:16)
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
