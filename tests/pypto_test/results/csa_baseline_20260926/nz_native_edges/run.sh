#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
source ../env-dsv4-0251rc1.sh
result_root="$PWD/tests/pypto_test/results/csa_baseline_20260926/nz_native_edges"
while read -r name variant batch history; do
    for mode in 0 2; do
        case_output="$result_root/${name}_mode${mode}"
        bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
            --batch "$batch" --history "$history" --variant "$variant" --weight-nz-mode "$mode" \
            --atomic-add 0 --graph --save-state > "${case_output}.log" 2>&1
    done
done <<'CASES'
short_b1 performance 1 255
long_b5 precision 5 32767
CASES
python "$result_root/compare.py"
