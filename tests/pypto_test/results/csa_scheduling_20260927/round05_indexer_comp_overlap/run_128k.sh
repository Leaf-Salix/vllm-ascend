#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
workspace=/data/pyptouser/qinchuanyu/pto-eager
for label in baseline_2dd51f15 round05_indexer_comp_overlap; do
    source_commit=2dd51f15
    if [[ "$label" == round05_indexer_comp_overlap ]]; then source_commit=07365e52; fi
    for kind in timing swimlane; do
        bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh "$label" 131072 16 "$kind" "$workspace/.cache/csa-scheduling-$source_commit"
    done
done
