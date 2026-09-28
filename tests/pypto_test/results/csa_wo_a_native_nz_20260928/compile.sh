#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_wo_a_native_nz_20260928"
source "$workspace/env-dsv4-0251rc1.sh"
python "$repo/tests/pypto_test/results/csa_maxlen_reuse_20260928/compile.py" \
    --source "$workspace/.cache/csa-wo-a-native-nz-3b27c7fd" --output "$root/compiled/candidate"
