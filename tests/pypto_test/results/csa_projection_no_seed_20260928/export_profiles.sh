#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
cd "$repo"
root=tests/pypto_test/results/csa_projection_no_seed_20260928
for history in 131072 8192; do
    case "$history" in
        131072) batches=(8) ;;
        8192) batches=(16) ;;
    esac
    for batch in "${batches[@]}"; do
        for backend in native pto; do
            python tests/pypto_test/offline_pd/run.py profile-export \
                --bank "tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank" \
                --output "$root/model/h$history/b$batch/$backend" --profile-ranks 0 --analyse-processes 4 \
                > "$root/model/h$history/b$batch/export_$backend.log" 2>&1
        done
    done
done
python tests/pypto_test/results/csa_source_split_ab_20260927/ordered/analyze_profile.py \
    --root "$root/model" --per-batch --cases 131072:8 8192:16
