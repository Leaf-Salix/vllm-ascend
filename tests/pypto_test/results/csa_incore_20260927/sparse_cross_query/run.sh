#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit via task-submit}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
probe_root="$repo/tests/pypto_test/results/csa_incore_20260927"
root="$probe_root/sparse_cross_query"
python - "$repo" "$root" <<'PY'
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(sys.argv[1]) / 'tests/pypto_test'))
from dsv4_csa_sparse_diagnostic import uniform_case
payload = uniform_case(9)
# T54 gives 2/3 queries per core. Queries 24..47 are entirely invalid:
# cores 0..5 therefore execute valid -> invalid -> valid without draining.
payload['seqused_kv'][4:8] = 0
payload['cmp_sparse_indices'][24:48] = -1
payload['expected'][24:48] = 0
torch.save(payload, Path(sys.argv[2]) / 'mixed_tail.pt')
PY
for name in standalone mixed_tail; do
    case_dir="$root/$name"
    mkdir -p "$case_dir/ascend"
    export ASCEND_PROCESS_LOG_PATH="$case_dir/ascend"
    cd "$case_dir"
    if [[ "$name" == standalone ]]; then
        input="$probe_root/sparse_pmu/native/native_sparse.pt"
    else
        input="$root/mixed_tail.pt"
    fi
    python "$repo/tests/pypto_test/dsv4_csa_sparse_diagnostic.py" \
        --input "$input" --output "$case_dir" --variant performance > run.log 2>&1
done
python - "$probe_root" <<'PY'
import sys,json
from pathlib import Path
import torch
root = Path(sys.argv[1])
baseline = torch.load(root / 'sparse_kv_early/gated_standalone/output.pt', map_location='cpu', weights_only=True)
actual = torch.load(root / 'sparse_cross_query/standalone/output.pt', map_location='cpu', weights_only=True)
assert torch.equal(actual, baseline), 'Cross-query pipeline must preserve the fixed-input output'
report = json.loads((root / 'sparse_cross_query/mixed_tail/report.json').read_text())
assert report['comparison']['status'] == 'PASS', report['comparison']
print('CROSS_QUERY_FIXED_INPUT_AND_MIXED_TAIL_BIT_EQUAL_PASS')
PY
cd "$repo"
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_cross_query 8192 40 timing
bash tests/pypto_test/results/csa_split_optimization_20260927/run_case.sh sparse_cross_query 8192 40 swimlane
