#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_scalar_medium 16 6 l0_db_control_t96
bash "$candidate_root/run_case.sh" stage1_explicit_l0_db_v1 16 6 l0_db_t96
