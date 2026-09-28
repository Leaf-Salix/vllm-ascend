#!/usr/bin/env bash
set -eo pipefail
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source "$workspace/env-dsv4-0251rc1.sh"
parent="$repo/tests/pypto_test/results/csa_score_balanced_sort_20260928"
root="$parent/b8_followup"
task=$(cat "$parent/b8_task.txt")
[[ "$(task-submit --status "$task")" == 'completed (exit=0)' ]]
python "$repo/tests/pypto_test/results/csa_short_score_sync_20260928/summarize.py" \
    --root "$root" --output "$root/h131072_b8/summary.json" --history 131072 --batch 8 \
    --iters 20 --windows 4 --task "$task" \
    --operator '8e176285 baseline; candidate adds balanced long leaves and 2560/3072 partial sorting'
python "$repo/tests/pypto_test/results/csa_score_balanced_leaves_20260928/incore.py" \
    --root "$root" --long-batch 8 --long-only --title '均衡分片及局部排序组合：长B8补充'
python - "$root/h131072_b8" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
source = json.loads((root / 'summary.json').read_text())
names = ('indexer_score_topk_native_pair_aic', 'indexer_score_topk_native_pair_aiv',
         'indexer_topk_query_merge', 'qk_pv_aic', 'qk_pv_aiv', 'merge_norm')
for side in source['measurements'].values():
    for window in side['worker_windows']:
        window['tasks'] = {name: window['tasks'][name] for name in names}
(root / 'evidence.json').write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n')
PY
