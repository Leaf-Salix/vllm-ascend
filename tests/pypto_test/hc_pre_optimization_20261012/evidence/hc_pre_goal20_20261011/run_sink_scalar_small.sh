#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" adopted_scalar_selective 4 6 sink_scalar_control_t24
bash "$candidate_root/run_case.sh" small_sink_scalar_v1 4 6 sink_scalar_t24
bash "$candidate_root/run_case.sh" adopted_scalar_selective 4 5 sink_scalar_control_t20
bash "$candidate_root/run_case.sh" small_sink_scalar_v1 4 5 sink_scalar_t20
