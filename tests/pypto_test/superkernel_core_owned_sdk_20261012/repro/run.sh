#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$p/artifacts/boundary_acceptance_20261012"
artifact="$p/artifacts/boundary_core_owned_sdk_20261012"
out="$artifact/retry1"
normal="$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py"
for part in old_before current_sdk old_after; do (
    if [[ "$part" == old_* ]]; then
        cd "$p/sdk-baseline/pypto"
        source "$p/sdk-baseline/env.sh"
        compiled="$base/baseline"
        modes=(plain)
    else
        cd "$p/pypto"
        source "$p/env.sh"
        compiled="$artifact/compiled"
        modes=(plain early)
    fi
    mkdir -p "$out/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/$part/ascend"
    python "$normal" run --compiled "$compiled" --output "$out/$part" \
        --calibration "$base/steady/calibration/results.json" --scalar-steps 256 --replays-per-graph 8 \
        --modes "${modes[@]}" --device "$TASK_DEVICE" > "$out/$part/run.log" 2>&1
    tail -7 "$out/$part/run.log"
); done
(
    cd "$p/pypto"
    source "$p/env.sh"
    mkdir -p "$out/diagnostics/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/diagnostics/ascend"
    python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
        tests/st/runtime/scheduling/test_local_task_boundary_mix.py \
        -k '(local_boundary_overlap and flat_wait) or (mixed_members_enter_prefix and default_events)' \
        --platform a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir "$out/diagnostics/kernels" \
        -q --junitxml "$out/diagnostics/results.xml" > "$out/diagnostics/run.log" 2>&1
    tail -4 "$out/diagnostics/run.log"
)
