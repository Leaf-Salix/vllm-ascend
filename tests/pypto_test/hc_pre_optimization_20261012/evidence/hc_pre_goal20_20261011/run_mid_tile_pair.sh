#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_balanced_acc 16 5 mid_tile_control_t80
bash "$candidate_root/run_case.sh" stream_mid_tile_acc_v1 16 5 mid_tile_t80
