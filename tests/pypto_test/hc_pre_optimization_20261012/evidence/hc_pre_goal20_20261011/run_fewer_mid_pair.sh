#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_scalar_medium 16 6 fewer_control_t96
bash "$candidate_root/run_case.sh" stage1_fewer_partials_db_v1 16 6 fewer_db_t96
bash "$candidate_root/run_case.sh" tail_schedule_incore_v1 16 6 fewer_schedule_t96
