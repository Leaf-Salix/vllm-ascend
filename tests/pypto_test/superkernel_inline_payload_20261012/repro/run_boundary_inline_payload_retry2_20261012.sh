#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="$p/artifacts/boundary_acceptance_20261012"
artifact="$p/artifacts/boundary_inline_payload_20261012"
out="$artifact/retry2"
normal="$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py"
probe="$p/pypto/tests/st/runtime/kernel/task_boundary_inline_payload_probe.py"
test -f "$artifact/binaries.json"
for part in old_before candidate old_after; do (
    if [[ "$part" == old_* ]]; then
        cd "$p/sdk-baseline/pypto"
        source "$p/sdk-baseline/env.sh"
        compiled="$base/baseline"
        modes=(plain)
    else
        cd "$p/pypto"
        source "$p/env.sh"
        compiled="$p/artifacts/boundary_normal_state_20261012/compiled"
        modes=(plain early)
    fi
    mkdir -p "$out/$part/ascend"
    export ASCEND_PROCESS_LOG_PATH="$out/$part/ascend"
    command=(python "$normal")
    if [[ "$part" == candidate ]]; then
        command=(python "$probe" run --root "$artifact")
    fi
    "${command[@]}" run --compiled "$compiled" --output "$out/$part" \
        --calibration "$base/steady/calibration/results.json" --scalar-steps 256 --replays-per-graph 8 \
        --modes "${modes[@]}" --device "$TASK_DEVICE" > "$out/$part/run.log" 2>&1
    tail -7 "$out/$part/run.log"
); done
