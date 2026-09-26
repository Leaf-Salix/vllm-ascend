#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
case_output="$PWD/tests/pypto_test/results/csa_baseline_20260926/perf_qproj_upstream"
bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
    --batch 16 --history 8192 --weight-nz-mode 2 --variant performance \
    --atomic-add 1 --deterministic-level 0 --timing-iters 20 --timing-warmup 5 \
    --timing-metadata reuse > "$case_output/run.log" 2>&1
