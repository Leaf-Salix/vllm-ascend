#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
for mode in 1 2; do
    case_output="$PWD/tests/pypto_test/results/csa_baseline_20260926/nz_native_b16_timing/following_mode${mode}"
    bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
        --batch 16 --history 8192 --weight-nz-mode "$mode" --variant performance \
        --atomic-add 1 --deterministic-level 0 --timing-iters 20 --timing-warmup 5 \
        --timing-metadata reuse --profile > "${case_output}.log" 2>&1
done
