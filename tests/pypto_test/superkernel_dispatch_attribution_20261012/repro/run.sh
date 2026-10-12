#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$p/artifacts/boundary_acceptance_20261012"
artifact="$p/artifacts/boundary_components_20261012"
normal="$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py"
probe="$p/pypto/tests/st/runtime/kernel/task_boundary_component_probe.py"
for part in old_before core cpu old_after; do (
    if [[ "$part" == old_* ]]; then
        cd "$p/sdk-baseline/pypto"
        source "$p/sdk-baseline/env.sh"
        compiled="$base/baseline"
        command=(python "$normal")
    else
        cd "$p/pypto"
        source "$p/env.sh"
        compiled="$p/artifacts/boundary_normal_state_20261012/compiled"
        command=(python "$probe" run --component "$part" --root "$artifact/$part")
    fi
    mkdir -p "$artifact/timing/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$artifact/timing/$part/ascend"
    "${command[@]}" run --compiled "$compiled" --output "$artifact/timing/$part" \
        --calibration "$base/steady/calibration/results.json" --scalar-steps 256 --replays-per-graph 8 \
        --device "$TASK_DEVICE" --modes plain > "$artifact/timing/$part/run.log" 2>&1
    tail -n 7 "$artifact/timing/$part/run.log"
); done
