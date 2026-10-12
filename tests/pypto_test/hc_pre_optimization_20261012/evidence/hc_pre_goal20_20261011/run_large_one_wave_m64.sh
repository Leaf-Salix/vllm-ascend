#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_balanced_acc 28 6 large_wave_control_m64
bash "$candidate_root/run_case.sh" stage1_large_one_wave_v1 28 6 large_wave_m64
