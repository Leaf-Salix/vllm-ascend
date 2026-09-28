#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
root="$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_score_balanced_sort_20260928"
source "$workspace/env-dsv4-0251rc1.sh"
python "$root/sort_case.py" --compile-only \
    --source "$workspace/.cache/csa-score-balanced-sort-8e176285" --output "$root/probe/candidate"
python "$root/sort_case.py" --compile-only \
    --source "$workspace/.cache/csa-score-balanced-8e176285" --output "$root/probe/baseline"
python "$root/compile.py"
echo 'ALL_COMPILE_PASS full CSA and both sorting probes; no device execution'
