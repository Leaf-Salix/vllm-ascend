#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
# Existing four-case evidence is retained. Check only the current combination
# on the long-history and large-batch representatives; no seven-case rerun.
for pair in '131072 16' '8192 40'; do
    read -r history batch <<< "$pair"
    bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh \
        indexer_fused_ws_restore "$history" "$batch" swimlane
done
