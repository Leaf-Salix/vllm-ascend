#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$p/artifacts/boundary_acceptance_20261012"
artifact="$p/artifacts/boundary_core_init_20261012"
out="$artifact/cached_failure_recheck"
normal="$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py"
for part in cached_after; do (
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
        if [[ "$part" == core_init ]]; then
            probe="$p/pypto/tests/st/runtime/kernel/task_boundary_core_init_probe.py"
            runtime_root="$artifact"
        else
            probe="$p/pypto/tests/st/runtime/kernel/task_boundary_contract_cache_probe.py"
            runtime_root="$p/artifacts/boundary_contract_cache_20261012"
        fi
        command=(python "$probe" run --root "$runtime_root")
    fi
    mkdir -p "$out/$part/ascend"
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export ASCEND_PROCESS_LOG_PATH="$out/$part/ascend"
    "${command[@]}" run --compiled "$compiled" --output "$out/$part" \
        --calibration "$base/steady/calibration/results.json" --scalar-steps 256 --replays-per-graph 8 \
        --modes "${modes[@]}" --device "$TASK_DEVICE" > "$out/$part/run.log" 2>&1
    tail -7 "$out/$part/run.log"
); done
