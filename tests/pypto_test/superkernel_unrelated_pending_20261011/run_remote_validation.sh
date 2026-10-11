#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?one queue-assigned device required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$p/pypto"
source "$p/env.sh"
export PYTHONPATH="$(python -c 'import os; print(":".join(p for p in os.environ["PYTHONPATH"].split(":") if not p.endswith(("/vllm-ascend", "/vllm"))))')"
python -m pytest tests/st/runtime/scheduling/test_local_task_boundary_remote_dependency.py \
  -q -n 1 \
  --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
  --kernels-dir "$p/artifacts/unrelated_pending_remote_device" \
  --junitxml "$p/artifacts/unrelated_pending_remote_device.xml"
