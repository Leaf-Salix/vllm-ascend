#!/usr/bin/env bash
set -eo pipefail
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
export ASCEND_PROCESS_LOG_PATH="$PWD/tests/pypto_test/results/csa_baseline_20260926/nz_layout_contract/ascend"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
exec python tests/pypto_test/dsv4_csa_weight_layout_probe.py --pypto-lib ../pypto-lib --output tests/pypto_test/results/csa_baseline_20260926/nz_layout_contract/report.json
