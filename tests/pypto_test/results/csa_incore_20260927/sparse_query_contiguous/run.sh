#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
probe_root="$repo/tests/pypto_test/results/csa_incore_20260927"
for name in standalone tail_b3; do
    case_dir="$probe_root/sparse_query_contiguous/$name"
    mkdir -p "$case_dir/ascend"
    export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
    cd "$case_dir"
    if [[ "$name" == standalone ]]; then
        args=(--input "$probe_root/sparse_pmu/native/native_sparse.pt")
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
baseline = torch.load(root / 'sparse_kv_early/gated_standalone/output.pt', map_location='cpu', weights_only=True)
actual = torch.load(root / 'sparse_query_contiguous/standalone/output.pt', map_location='cpu', weights_only=True)
assert torch.equal(actual, baseline), 'Query ownership must preserve the fixed-input output'
print('CONTIGUOUS_QUERY_BIT_EQUAL_PASS')
PY
cd "$repo"
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_query_contiguous 8192 40 timing
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_query_contiguous 8192 40 swimlane
