#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export PYTHONPATH="$repo/tests/pypto_test:${PYTHONPATH:-}"
root="$repo/tests/pypto_test/results/csa_forward_event_prewarm_20260928"
cd "$root"
for mode in 0 1; do
    python "$root/probe.py" --device "$TASK_DEVICE" --mode "$mode" --output "$root/mode${mode}.json" \
        > "$root/mode${mode}.log" 2>&1
done
