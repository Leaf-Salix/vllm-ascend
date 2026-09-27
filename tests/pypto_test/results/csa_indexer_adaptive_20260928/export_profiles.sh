#!/usr/bin/env bash
# CPU-only export/analysis after the model task has completed.
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
root="$repo/tests/pypto_test/results/csa_indexer_adaptive_20260928/model"
cd "$repo"
for history in 131072 8192; do
    case "$history" in
        131072) batches=(4 8 16) ;;
        8192) batches=(16 24 32 40) ;;
    esac
    for batch in "${batches[@]}"; do
        for backend in native pto; do
            python tests/pypto_test/offline_pd/run.py profile-export \
                --bank "tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank" \
                --output "$root/h$history/b$batch/$backend" --profile-ranks 0 --analyse-processes 4 \
                > "$root/h$history/b$batch/export_$backend.log" 2>&1
        done
    done
done
python tests/pypto_test/results/csa_source_split_ab_20260927/ordered/analyze_profile.py \
    --root "$root" --per-batch --cases 131072:4 131072:8 131072:16 8192:16 8192:24 8192:32 8192:40
