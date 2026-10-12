#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
a="$p/artifacts/boundary_compact_dispatch_20261012"
out="$a/retry2"
for part in normal_before compact normal_after; do (
    if [[ "$part" == compact ]]; then
        cd "$p/pypto-compact-dispatch-20261012"
        source "$p/sdk-compact-dispatch-20261012/env.sh"
        cmd=(python "$a/repro/driver.py" run --root "$a")
        compiled="$a/compiled"
    else
        cd "$p/pypto"
        source "$p/env.sh"
        cmd=(python "$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py" run)
        compiled="$p/artifacts/boundary_core_owned_sdk_20261012/compiled"
    fi
    mkdir -p "$out/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/$part/ascend"
    "${cmd[@]}" --compiled "$compiled" --output "$out/$part" --calibration "$a/calibration_5us.json" \
        --scalar-steps 256 --replays-per-graph 8 --modes plain early --device "$TASK_DEVICE" > "$out/$part/run.log" 2>&1
    tail -7 "$out/$part/run.log"
); done
