#!/usr/bin/env bash
set -eo pipefail
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_b24_integrated_20260929
bash "$root/run_swimlane.sh"
bash "$root/run_graph.sh"
