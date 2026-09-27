#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
root="$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_split_optimization_20260927"
case "${1:-gm}" in
    gm) selector=acc_to_gm_dequant_relu; result_dir="$root/fixpipe" ;;
    mat) selector=acc_to_mat_dequant_relu_then_matmul; result_dir="$root/fixpipe_mat" ;;
    *) exit 2 ;;
esac
source "$workspace/env-dsv4-0251rc1.sh"
cd "$workspace/pypto"
source .claude/skills/testing/load-env.sh
mkdir -p "$result_dir"
python -m pytest -q tests/st/runtime/ops/test_fixpipe_epilogue.py \
    -k "$selector" --platform a2a3 --device "$TASK_DEVICE" \
    > "$result_dir/device.log" 2>&1
