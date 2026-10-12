#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
a="$p/artifacts/boundary_compact_sdk_20261012"
out="$a/retry1"
for part in old_before current_sdk old_after; do (
    if [[ "$part" == old_* ]]; then
        cd "$p/sdk-baseline/pypto"
        source "$p/sdk-baseline/env.sh"
        compiled="$p/artifacts/boundary_acceptance_20261012/baseline"
        modes=(plain)
    else
        cd "$p/pypto"
        source "$p/env.sh"
        compiled="$a/compiled"
        modes=(plain early)
    fi
    mkdir -p "$out/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/$part/ascend"
    python "$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py" run --compiled "$compiled" \
        --output "$out/$part" --calibration "$p/artifacts/boundary_compact_dispatch_20261012/calibration_5us.json" \
        --scalar-steps 256 --replays-per-graph 8 --modes "${modes[@]}" --device "$TASK_DEVICE" > "$out/$part/run.log" 2>&1
    tail -7 "$out/$part/run.log"
); done
