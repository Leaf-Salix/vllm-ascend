#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_balanced_acc 20 5 large_wave_control_t100
bash "$candidate_root/run_case.sh" stage1_large_one_wave_v1 20 5 large_wave_t100
bash "$candidate_root/run_case.sh" adopted_balanced_acc 20 6 large_wave_control_t120
bash "$candidate_root/run_case.sh" stage1_large_one_wave_v1 20 6 large_wave_b20s6
bash "$candidate_root/run_case.sh" stage1_large_one_wave_v1 24 5 large_wave_b24s5
