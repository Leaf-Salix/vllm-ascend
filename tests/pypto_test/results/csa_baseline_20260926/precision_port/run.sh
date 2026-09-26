#!/usr/bin/env bash
set -eo pipefail
repo_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1
phase="${1:?指定 baseline 或 candidate}"
cd "$repo_root"
for batch in 16 1 40; do
    history=8192
    timing=(--timing-iters 0)
    extra=()
    if [[ "$batch" == 16 ]]; then
        timing=(--timing-iters 20 --timing-warmup 5 --timing-metadata reuse)
        if [[ "$phase" != baseline ]]; then
            extra=(--graph)
        fi
    elif [[ "$batch" == 1 ]]; then
        history=255
    fi
    case_output="$repo_root/tests/pypto_test/results/csa_baseline_20260926/precision_port/$phase/b$batch"
    mkdir -p "$case_output"
    bash tests/pypto_test/run_csa_single_layer.sh "$case_output" \
        --batch "$batch" --history "$history" --weight-nz-mode 2 --variant precision \
        --atomic-add 0 --deterministic-level 1 --save-state \
        "${timing[@]}" "${extra[@]}" > "$case_output/run.log" 2>&1
done
