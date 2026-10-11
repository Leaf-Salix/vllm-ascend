#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$p/pypto/python:$p/simpler/python:$p/simpler"
python tests/st/runtime/scheduling/task_boundary_allocated_event_probe.py --event-id exhausted --compile \
  --compiled "$p/artifacts/blocking_event_probe_compiled_20261012" \
  > "$p/artifacts/blocking_event_probe_compile_20261012.log" 2>&1
