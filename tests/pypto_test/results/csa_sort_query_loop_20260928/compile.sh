#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_sort_query_loop_20260928"
source "$workspace/env-dsv4-0251rc1.sh"
python "$repo/tests/pypto_test/results/csa_maxlen_reuse_20260928/compile.py" \
    --source "$workspace/.cache/csa-sort-query-loop-d0addbb4" --output "$root/compiled/candidate"
