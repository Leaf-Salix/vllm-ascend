#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
probe_root="$repo/tests/pypto_test/results/csa_incore_20260927"
for name in gated_standalone gated_tail_b3; do
    case_dir="$probe_root/sparse_kv_early/$name"
    mkdir -p "$case_dir/ascend"
    export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
    cd "$case_dir"
    if [[ "$name" == gated_standalone ]]; then
        args=(--input "$probe_root/sparse_pmu/native/native_sparse.pt" --pmu 2)
    else
        args=(--synthetic-batch 3)
    fi
    python "$repo/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" \
        "${args[@]}" --output "$case_dir" --variant performance > run.log 2>&1
done
python - "$probe_root" <<'PY'
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
baseline = torch.load(root / 'sparse_pmu/standalone_pipe/output.pt', map_location='cpu', weights_only=True)
actual = torch.load(root / 'sparse_kv_early/gated_standalone/output.pt', map_location='cpu', weights_only=True)
assert torch.equal(actual, baseline), 'Event dispatch must preserve the fixed-input output'
print('GATED_FIXED_INPUT_BIT_EQUAL_PASS')
PY
cd "$repo"
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_kv_early_gated 8192 40 timing
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_kv_early_gated 8192 40 swimlane
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_kv_early_gated 8192 16 timing
