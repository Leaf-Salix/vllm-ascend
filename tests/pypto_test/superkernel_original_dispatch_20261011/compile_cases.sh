#!/usr/bin/env bash
set -eo pipefail
private_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$private_root/pypto"
source "$private_root/env.sh"
export PYTHONPATH="$(python -c 'import os; print(":".join(p for p in os.environ["PYTHONPATH"].split(":") if not p.endswith(("/vllm-ascend", "/vllm"))))')"
python -m pytest tests/st/runtime/scheduling/test_task_boundary_statements.py \
    tests/st/runtime/scheduling/test_local_task_boundary_overlap.py \
    -q -n "$PYPTO_TEST_JOBS" --platform a2a3 --codegen-only --save-kernels \
    --kernels-dir "$private_root/artifacts/original_dispatch_local_codegen" \
    > "$private_root/artifacts/original_dispatch_local_codegen.log" 2>&1
python tests/st/runtime/scheduling/task_boundary_no_successor_demo.py --compile \
    --compiled "$private_root/artifacts/original_dispatch_no_successor_compile" \
    > "$private_root/artifacts/original_dispatch_no_successor_compile.log" 2>&1
printf '%s\n' 'STAGE: local boundary cases compiled'
cd "$private_root/vllm-ascend"
source "$private_root/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
for variant in baseline all; do
    python tests/pypto_test/tmr_csa_full_128k_b16_compile.py --variant "$variant" \
        --output "$private_root/artifacts/csa_original_dispatch_compile" \
        > "$private_root/artifacts/csa_original_dispatch_compile_$variant.log" 2>&1
done
printf '%s\n' 'STAGE: baseline and all CSA compiled'
