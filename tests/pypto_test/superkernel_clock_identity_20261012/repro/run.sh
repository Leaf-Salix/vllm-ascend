#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
artifact="$p/artifacts/boundary_clock_identity_20261012"
out="$artifact/retry1"
normal="$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py"
compiled="$p/artifacts/boundary_compact_sdk_20261012/compiled"
calibration="$artifact/calibration_5us.json"
for part in normal_before probe normal_after; do (
    cd "$p/pypto"
    source "$p/env.sh"
    mkdir -p "$out/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/$part/ascend"
    common=(run --compiled "$compiled" --output "$out/$part" --calibration "$calibration"
        --scalar-steps 256 --replays-per-graph 8 --modes plain early --device "$TASK_DEVICE")
    if [[ "$part" == probe ]]; then
        python "$artifact/repro/driver.py" run --root "$artifact" "${common[@]}" > "$out/$part/run.log" 2>&1
    else
        python "$normal" "${common[@]}" > "$out/$part/run.log" 2>&1
    fi
    tail -7 "$out/$part/run.log"
); done
