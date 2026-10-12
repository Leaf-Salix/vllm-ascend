#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
artifact="$p/artifacts/boundary_doorbell_probe_20261012"
out="$artifact/retry2"
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
(
    cd "$p/pypto"
    source "$p/env.sh"
    mkdir -p "$out/diagnostics/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/diagnostics/ascend"
    python "$artifact/repro/driver.py" diagnostics --root "$artifact" tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
        tests/st/runtime/scheduling/test_local_task_boundary_mix.py \
        -k '(local_boundary_overlap and flat_wait) or (mixed_members_enter_prefix and default_events)' \
        --platform a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir "$out/diagnostics/kernels" \
        -q --junitxml "$out/diagnostics/results.xml" > "$out/diagnostics/run.log" 2>&1
    tail -4 "$out/diagnostics/run.log"
)
