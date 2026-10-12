#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_balanced_acc 2 6 sink_compact_control_t12
bash "$candidate_root/run_case.sh" sink_compact_block_v3 2 6 sink_compact_t12
bash "$candidate_root/run_case.sh" adopted_balanced_acc 28 6 sink_compact_control_t168
bash "$candidate_root/run_case.sh" sink_compact_block_v3 28 6 sink_compact_t168
