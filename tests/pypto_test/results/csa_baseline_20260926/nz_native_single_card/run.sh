#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
bash tests/pypto_test/results/csa_baseline_20260926/nz_layout_contract/run.sh
for variant in performance precision; do
    for mode in 0 2; do
        case_output="$PWD/tests/pypto_test/results/csa_baseline_20260926/nz_native_single_card/${variant}_mode${mode}"
        bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
            --batch 4 --history 8192 --weight-nz-mode "$mode" --variant "$variant" \
            --atomic-add 0 --graph --save-state
    done
done
