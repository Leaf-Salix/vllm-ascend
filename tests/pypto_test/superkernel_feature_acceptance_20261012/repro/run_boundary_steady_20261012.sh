#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$p/artifacts/boundary_acceptance_20261012"
out="$base/steady"
driver="$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py"
run_part() (
    name="$1"
    environment="$2"
    action="$3"
    compiled="$4"
    shift 4
    cd "$environment/pypto"
    source "$environment/env.sh"
    mkdir -p "$out/$name/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/$name/ascend"
    python "$driver" "$action" --device "$TASK_DEVICE" --compiled "$compiled" \
        --output "$out/$name" --calibration "$out/calibration/results.json" \
        --scalar-steps 256 --replays-per-graph 8 "$@" > "$out/$name/run.log" 2>&1
    tail -8 "$out/$name/run.log"
)
run_part calibration "$p" calibrate "$base/current"
run_part before "$p/sdk-baseline" run "$base/baseline" --modes plain
run_part current "$p" run "$base/current" --modes plain early
run_part after "$p/sdk-baseline" run "$base/baseline" --modes plain
(
    cd "$p/pypto"
    source "$p/env.sh"
    mkdir -p "$base/function_v2/ascend"
    export ASCEND_PROCESS_LOG_PATH="$base/function_v2/ascend"
    python tests/st/runtime/scheduling/task_boundary_compute_marker.py test --root "$base/function_v2" \
        tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
        -k 'local_boundary_overlap and flat_wait' --platform a2a3 --device "$TASK_DEVICE" \
        --save-kernels --kernels-dir "$base/function_v2/kernels" -q \
        --junitxml "$base/function_v2/results.xml" > "$base/function_v2/run.log" 2>&1
    tail -5 "$base/function_v2/run.log"
)
