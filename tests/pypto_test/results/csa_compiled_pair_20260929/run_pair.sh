#!/usr/bin/env bash
set -eo pipefail
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_pair_20260929
# First case validates PTO compilation; stop immediately on failure.
# All four processes share the task's allocated device.
bash "$root/run_case.sh" pto 131072 16
bash "$root/run_case.sh" native 131072 16
bash "$root/run_case.sh" native 8192 24
bash "$root/run_case.sh" pto 8192 24
