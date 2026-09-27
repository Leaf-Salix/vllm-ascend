#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
cd "$repo"
root=tests/pypto_test/results/csa_qa_matrix_20260928
histories=(131072 8192)
if [[ $# -gt 0 ]]; then histories=("$1"); fi
backends=(native pto)
if [[ $# -gt 1 ]]; then backends=("$2"); fi
cases=()
for history in "${histories[@]}"; do
    case "$history" in
        131072) batches=(4 8 16) ;;
        8192) batches=(16 24 32 40) ;;
    esac
    for batch in "${batches[@]}"; do
        cases+=("$history:$batch")
        for backend in "${backends[@]}"; do
            python tests/pypto_test/offline_pd/run.py profile-export \
                --bank "tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank" \
                --output "$root/model/h$history/b$batch/$backend" --profile-ranks 0 --analyse-processes 4 \
                > "$root/model/h$history/b$batch/export_$backend.log" 2>&1
        done
    done
done
if [[ $# -gt 1 ]]; then exit 0; fi
python tests/pypto_test/results/csa_source_split_ab_20260927/ordered/analyze_profile.py \
    --root "$root/model" --per-batch --cases "${cases[@]}"
python "$root/report.py" model
