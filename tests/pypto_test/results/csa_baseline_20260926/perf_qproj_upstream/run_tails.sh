#!/usr/bin/env bash
set -eo pipefail
cd /data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
root="$PWD/tests/pypto_test/results/csa_baseline_20260926/perf_qproj_upstream"
while read -r name batch history; do
    case_output="$root/$name"
    bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
        --batch "$batch" --history "$history" --weight-nz-mode 2 --variant performance \
        --atomic-add 1 --deterministic-level 0 > "$root/${name}.log" 2>&1
done <<'CASES'
tail_b1 1 255
tail_b40 40 8192
CASES
