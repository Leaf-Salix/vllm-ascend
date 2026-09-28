#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
root="$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_maxlen_reuse_20260928"
source "$workspace/env-dsv4-0251rc1.sh"
python "$root/compile.py" --source "$workspace/.cache/csa-maxlen-reuse-e33d842a" --output "$root/compiled/candidate"
python "$root/compile.py" --source "$workspace/.cache/csa-cann92-baseline-e33d842a" --output "$root/compiled/baseline"
python "$root/length_graph_probe.py" --source "$workspace/.cache/csa-maxlen-reuse-e33d842a" \
    --output "$root/length_probe" --compile-only
echo 'ALL_COMPILE_PASS CANN 9.2 baseline and max-length reuse candidate'
