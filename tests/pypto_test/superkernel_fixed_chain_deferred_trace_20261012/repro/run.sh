#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$p/artifacts/boundary_dispatch_deferred_trace_20261012"
cd "$p/pypto"
source "$p/env.sh"
mkdir -p "$root/ascend"
export ASCEND_PROCESS_LOG_PATH="$root/ascend"
python tests/st/runtime/kernel/task_boundary_dispatch_deferred_trace.py run --root "$root" \
    --device "$TASK_DEVICE" > "$root/run.log" 2>&1
python tests/st/runtime/kernel/task_boundary_dispatch_analysis.py "$root/results.json" > "$root/analysis.log"
tail -3 "$root/run.log"
