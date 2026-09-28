#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_sort_query_loop_20260928"
exec bash "$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928/run_layer.sh" \
    "$root" "$workspace/.cache/csa-sort-query-loop-d0addbb4" 16 \
    "$workspace/.cache/csa-cann92-baseline-e33d842a"
