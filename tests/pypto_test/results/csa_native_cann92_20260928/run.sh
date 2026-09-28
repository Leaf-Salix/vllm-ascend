#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_native_cann92_20260928
# Start with the long representative; expand only after actual compatibility.
for spec in ${CSA_CANN_CASES:-131072:16 8192:16}; do
    history="${spec%:*}"
    batch="${spec#*:}"
    for version in cann92 cann90; do
        (
            source "$root/env.sh" "$version"
            out="$root/h${history}_b${batch}/$version"
            test ! -e "$out/report.json"
            mkdir -p "$out/ascend" "$out/ascend_cache"
            export ASCEND_PROCESS_LOG_PATH="$out/ascend"
            export ASCEND_CACHE_PATH="$out/ascend_cache"
            cd "$out"
            python "$root/native_case.py" \
                --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
                --output "$out" --device "$TASK_DEVICE" --batch "$batch" --history "$history" \
                --layer-index 4 --weight-nz-mode 2 --seed 1024 --deterministic-level 0 \
                --native-profile-only --timing-iters 20 --timing-warmup 5 > "$out/run.log" 2>&1
        )
    done
done
