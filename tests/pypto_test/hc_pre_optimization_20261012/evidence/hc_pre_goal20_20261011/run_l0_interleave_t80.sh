#!/usr/bin/env bash
set -eo pipefail
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh adopted_l0_interleave_exact96 16 5 l0_interleave_t80_control
bash /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011/run_case.sh l0_interleave_medium96_v2 16 5 l0_interleave_t80
