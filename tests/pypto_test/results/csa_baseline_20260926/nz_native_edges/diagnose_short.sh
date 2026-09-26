#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source ../env-dsv4-0251rc1.sh
case_output="$PWD/tests/pypto_test/results/csa_baseline_20260926/nz_native_edges/short_b1_precision_mode2"
bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
    --batch 1 --history 255 --variant precision --weight-nz-mode 2 --atomic-add 0 \
    --save-state --save-case > "${case_output}.log" 2>&1
python "$PWD/tests/pypto_test/results/csa_baseline_20260926/nz_native_edges/compare.py" --short-diagnostic
