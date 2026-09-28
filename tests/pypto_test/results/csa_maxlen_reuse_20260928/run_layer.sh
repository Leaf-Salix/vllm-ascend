#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_maxlen_reuse_20260928"
rg -q '^ALL_COMPILE_PASS ' "$root/compile.log"
source "$workspace/env-dsv4-0251rc1.sh"
[[ "$ASCEND_HOME_PATH" == */cann-9.2.0-beta.2 ]]
mkdir -p "$root/length_probe"
cd "$root/length_probe"
python "$root/length_graph_probe.py" --source "$workspace/.cache/csa-maxlen-reuse-e33d842a" \
    --output "$root/length_probe" > "$root/length_probe/run.log" 2>&1
exec bash "$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928/run_layer.sh" \
    "$root" "$workspace/.cache/csa-maxlen-reuse-e33d842a" 16 \
    "$workspace/.cache/csa-cann92-baseline-e33d842a"
