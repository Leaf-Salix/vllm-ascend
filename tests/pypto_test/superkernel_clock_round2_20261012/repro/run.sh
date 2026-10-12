#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
a="$p/artifacts/boundary_clock_round2_20261012"
for part in normal_before cycle_gate clock_identity normal_after; do (
    cd "$p/pypto"
    source "$p/env.sh"
    mkdir -p "$a/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$a/$part/ascend"
    common=(run --compiled "$p/artifacts/boundary_compact_sdk_20261012/compiled"
        --output "$a/$part" --calibration "$p/artifacts/boundary_cycle_gate_20261012/calibration_5us.json"
        --scalar-steps 256 --replays-per-graph 8 --modes plain early --device "$TASK_DEVICE")
    if [[ "$part" == normal_* ]]; then
        python "$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py" "${common[@]}" > "$a/$part/run.log" 2>&1
    else
        candidate="$p/artifacts/boundary_${part}_20261012"
        python "$candidate/repro/driver.py" run --root "$candidate" "${common[@]}" > "$a/$part/run.log" 2>&1
    fi
    tail -7 "$a/$part/run.log"
); done
(
    cd "$p/pypto"
    source "$p/env.sh"
    diagnostic="$p/artifacts/boundary_clock_diagnostic_20261012"
    mkdir -p "$a/diagnostic/ascend"
    export ASCEND_PROCESS_LOG_PATH="$a/diagnostic/ascend"
    python "$diagnostic/repro/profile_check.py" --root "$diagnostic" --output "$a/diagnostic/dfx" > "$a/diagnostic/run.log" 2>&1
    tail -5 "$a/diagnostic/run.log"
)
