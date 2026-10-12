#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_large_pipeline 20 5 large_scalar_control_m48
bash "$candidate_root/run_case.sh" large_scalar_mix_v1 20 5 large_scalar_m48
