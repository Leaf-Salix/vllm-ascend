#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for variant in performance precision; do
    case_output="$PWD/tests/pypto_test/results/csa_baseline_20260926/nz_native_padding/${variant}"
    bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
        --batch 4 --history 4095 --variant "$variant" --weight-nz-mode 2 \
        --atomic-add 0 --padding-graph > "${case_output}.log" 2>&1
done
