#!/usr/bin/env bash
set -eo pipefail
private_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK_DEVICE:?single-card task-submit assignment is required}"
[[ "$TASK_DEVICE" =~ ^[0-9]+$ ]]
cd "$private_root/pypto"
source "$private_root/env.sh"
export PYTHONPATH="$(python -c 'import os; print(":".join(p for p in os.environ["PYTHONPATH"].split(":") if not p.endswith(("/vllm-ascend", "/vllm"))))')"
python -m pytest tests/st/runtime/scheduling/test_task_boundary_statements.py \
    tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
    -q -n 1 --platform a2a3 --device "$TASK_DEVICE" --save-kernels \
    --kernels-dir "$private_root/artifacts/original_dispatch_local_device" \
    --junitxml "$private_root/artifacts/original_dispatch_local_device.xml"
python tests/st/runtime/scheduling/task_boundary_no_successor_demo.py \
    --compiled "$private_root/artifacts/original_dispatch_no_successor_compile" \
    --device "$TASK_DEVICE" --output "$private_root/artifacts/original_dispatch_no_successor_device"
