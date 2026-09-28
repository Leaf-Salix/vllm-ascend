#!/usr/bin/env bash
set -eo pipefail
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_b24_integrated_20260929
for side in baseline candidate; do
    rg -q COMPILE_PASS "$root/compile_${side}.json"
done
bash "$root/run_case.sh" baseline pto 131072 24
bash "$root/run_case.sh" candidate native 131072 24
bash "$root/run_case.sh" candidate pto 131072 24
bash "$root/run_swimlane.sh"
bash "$root/run_graph.sh"
