#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with one device}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
parent="$repo/tests/pypto_test/results/csa_score_balanced_sort_20260928"
root="$parent/b8_followup"
rg -q '^ALL_COMPILE_PASS ' "$parent/compile.log"
python3 - "$parent" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for case in ('h131072_b16', 'h8192_b16'):
    assert json.loads((root / case / 'summary.json').read_text())['status'] == 'PASS'
assert json.loads((root / 'probe/candidate/report.json').read_text())['baseline_bit_comparison'] == 'PASS'
PY
mkdir -p "$root"
ln -s ../compile.log "$root/compile.log"
exec bash "$repo/tests/pypto_test/results/csa_score_key_l1_pair_20260928/run_layer.sh" \
    "$root" "$workspace/.cache/csa-score-balanced-sort-8e176285" 8 \
    "$workspace/.cache/csa-key-prefetch-final-8e176285" '131072:8'
