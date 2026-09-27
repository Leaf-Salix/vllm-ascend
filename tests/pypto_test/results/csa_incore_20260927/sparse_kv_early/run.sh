#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
probe_root="$repo/tests/pypto_test/results/csa_incore_20260927"
case_dir="$probe_root/sparse_kv_early/standalone"
mkdir -p "$case_dir/ascend"
export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
cd "$case_dir"
python "$repo/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" \
    --input "$probe_root/sparse_pmu/native/native_sparse.pt" --output "$case_dir" \
    --variant performance --pmu 2 > "$case_dir/run.log" 2>&1
python - "$probe_root" <<'PY'
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
baseline = torch.load(root / 'sparse_pmu/standalone_pipe/output.pt', map_location='cpu', weights_only=True)
actual = torch.load(root / 'sparse_kv_early/standalone/output.pt', map_location='cpu', weights_only=True)
assert torch.equal(actual, baseline), 'Changing only the event order must preserve the fixed-input output'
print('FIXED_INPUT_BIT_EQUAL_PASS')
PY
cd "$repo"
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_kv_early 8192 40 timing
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_kv_early 8192 40 swimlane
