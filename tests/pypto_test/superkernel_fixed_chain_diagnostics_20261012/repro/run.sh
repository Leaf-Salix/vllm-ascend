#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
out="$p/artifacts/boundary_chain_trace_20261012/retry7"
mkdir -p "$out/ascend"
export ASCEND_PROCESS_LOG_PATH="$out/ascend"
python tests/st/runtime/kernel/task_boundary_chain_trace.py run --root "$out" \
    --runtime-root "$p/artifacts/boundary_compact_init_20261012" --device "$TASK_DEVICE" > "$out/run.log" 2>&1
cat "$out/run.log"
