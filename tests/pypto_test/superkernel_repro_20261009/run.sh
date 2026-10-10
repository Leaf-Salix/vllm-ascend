#!/usr/bin/env bash
# Run only via task-submit --device auto; compile before reserving a device.
set -eE -o pipefail
trap 'printf "Repro failed at %s:%s (exit %s)\n" "${BASH_SOURCE[0]}" "$LINENO" "$?" >&2' ERR
repro_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$repro_dir/../../.." && pwd)
source "$repo_dir/../env-dsv4-0251rc1.sh"
if [[ ! ${TASK_DEVICE:-} =~ ^[0-9]+$ ]]; then
    printf 'A single card allocated by task-submit is required.\n' >&2
    exit 2
fi
output_dir="$repo_dir/tests/pypto_test/results/native_superkernel_repro_20261009"
while (( $# )); do
    case "$1" in
        --device)
            [[ ${2:-} == "$TASK_DEVICE" ]] || exit 2
            shift 2 ;;
        *) output_dir="$1"; shift ;;
    esac
done
binary="$output_dir/build/native_early_start"
if [[ ! -x "$binary" ]]; then
    printf 'Missing executable: %s (run build.sh first)\n' "$binary" >&2
    exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE"
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_PROCESS_LOG_PATH="$output_dir/cann_log"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
printf 'physical_device=%s CANN=%s\n' "$TASK_DEVICE" "$ASCEND_HOME_PATH" > "$output_dir/environment.txt"
# 5 warmups + 20 measured replays. Same arithmetic/workload in all four arms.
"$binary" 0 0 0 0 32 256 20 > "$output_dir/01_graph.jsonl" 2> "$output_dir/01_graph.stderr"
"$binary" 0 1 0 0 32 256 20 > "$output_dir/02_sk_early_off.jsonl" 2> "$output_dir/02_sk_early_off.stderr"
"$binary" 0 1 1 1 32 256 20 > "$output_dir/03_sk_wait_first.jsonl" 2> "$output_dir/03_sk_wait_first.stderr"
"$binary" 0 1 1 0 32 256 20 > "$output_dir/04_sk_wait_after_scalar.jsonl" 2> "$output_dir/04_sk_wait_after_scalar.stderr"
cat "$output_dir/01_graph.jsonl" "$output_dir/02_sk_early_off.jsonl" \
    "$output_dir/03_sk_wait_first.jsonl" "$output_dir/04_sk_wait_after_scalar.jsonl"
