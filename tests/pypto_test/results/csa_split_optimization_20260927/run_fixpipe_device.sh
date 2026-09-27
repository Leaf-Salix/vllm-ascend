#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
root="$workspace/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_split_optimization_20260927"
source "$workspace/env-dsv4-0251rc1.sh"
cd "$workspace/pypto"
source .claude/skills/testing/load-env.sh
mkdir -p "$root/fixpipe"
python -m pytest -q tests/st/runtime/ops/test_fixpipe_epilogue.py \
    -k acc_to_gm_dequant_relu --platform a2a3 --device "$TASK_DEVICE" \
    > "$root/fixpipe/device.log" 2>&1
