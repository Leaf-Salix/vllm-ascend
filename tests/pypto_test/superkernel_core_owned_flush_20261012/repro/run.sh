#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$p/artifacts/boundary_acceptance_20261012"
artifact="$p/artifacts/boundary_core_owned_flush_20261012"
out="$artifact/retry1"
normal="$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py"
for part in old_before cached_before core_owned_flush cached_after old_after; do (
    if [[ "$part" == old_* ]]; then
        cd "$p/sdk-baseline/pypto"
        source "$p/sdk-baseline/env.sh"
        compiled="$base/baseline"
        modes=(plain)
        command=(python "$normal")
    else
        cd "$p/pypto"
        source "$p/env.sh"
        compiled="$p/artifacts/boundary_compiler_proof_20261012/compiled"
        modes=(plain early)
        if [[ "$part" == core_owned_flush ]]; then
            probe="$p/pypto/tests/st/runtime/kernel/task_boundary_core_owned_flush_probe.py"
            runtime_root="$artifact"
        else
            probe="$p/pypto/tests/st/runtime/kernel/task_boundary_contract_cache_probe.py"
            runtime_root="$p/artifacts/boundary_contract_cache_20261012"
        fi
        command=(python "$probe" run --root "$runtime_root")
    fi
    mkdir -p "$out/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/$part/ascend"
    "${command[@]}" run --compiled "$compiled" --output "$out/$part" \
        --calibration "$base/steady/calibration/results.json" --scalar-steps 256 --replays-per-graph 8 \
        --modes "${modes[@]}" --device "$TASK_DEVICE" > "$out/$part/run.log" 2>&1
    tail -7 "$out/$part/run.log"
); done

(
    cd "$p/pypto"
    source "$p/env.sh"
    probe="$p/pypto/tests/st/runtime/kernel/task_boundary_core_owned_flush_probe.py"
    mkdir -p "$out/diagnostics/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/diagnostics/ascend"
    python "$probe" diagnostics --root "$artifact" \
        tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
        tests/st/runtime/scheduling/test_local_task_boundary_mix.py \
        -k '(local_boundary_overlap and flat_wait) or (mixed_members_enter_prefix and default_events)' \
        --platform a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir "$out/diagnostics/kernels" \
        -q --junitxml "$out/diagnostics/results.xml" > "$out/diagnostics/run.log" 2>&1
    tail -4 "$out/diagnostics/run.log"
)
