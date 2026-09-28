#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit with 16 devices}"
IFS=',' read -r -a stage_devices <<< "$TASK_DEVICE"
[[ "${#stage_devices[@]}" == 16 ]]
# Queue status was checked before submission; nested task-submit is forbidden.
# The four completed single-card state reports below remain the on-task input gate.
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
source_repo="$workspace/.cache/csa-key-prefetch-final-8e176285"
source "$workspace/env-dsv4-0251rc1.sh"
python - "$repo/tests/pypto_test/results" <<'CHECK'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for family, long_batch in (('csa_score_key_l1_pair_20260928', 4), ('csa_score_key_l1_only_20260928', 8)):
    for history, batch in ((131072, long_batch), (8192, 16)):
        data = json.loads((root / family / f'h{history}_b{batch}/evidence.json').read_text())
        assert data['status'] == data['graph_status'] == 'PASS' and not data['errors']
        assert len(data['checks']) == 8 and all(x['status'] == 'PASS' for x in data['checks'].values())
CHECK
rg -q '^COMPILE_PASS combined ' "$repo/tests/pypto_test/results/csa_score_key_l1_only_20260928/combined_compile.log"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$workspace/.cache/migration-v0.25.1rc1/vllm:$source_repo:$source_repo/tests/pypto_test:${PYTHONPATH:-}"
export PTO_CSA_VARIANT=performance
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export HCCL_DETERMINISTIC=false
for history in 131072 8192; do
    out="$repo/tests/pypto_test/results/csa_key_prefetch_final_20260928/model/h$history"
    bank="$repo/tests/pypto_test/results/release_offline_pd_20260923/h${history}_bank"
    case "$history" in
        131072) batches=(4 8); budget=256; port=31941; backends=(pto native) ;;
        8192) batches=(16); budget=400; port=31961; backends=(native pto) ;;
    esac
    mkdir -p "$out"
    cd "$out"
    for backend in "${backends[@]}"; do
        [[ ! -e "$out/${backend}_launch.log" ]]
        mkdir -p "$out/$backend/ascend"
        export ASCEND_PROCESS_LOG_PATH="$out/$backend/ascend"
        python "$source_repo/tests/pypto_test/offline_pd/run.py" performance \
            --bank "$bank" --output "$out/$backend" --backend "$backend" --port "$port" \
            --batch 40 --sweep-batches "${batches[@]}" --max-num-batched-tokens "$budget" \
            --decode-tokens 128 --forward-host-diagnostics --weight-nz-mode 2 --graph-mode full_decode_only \
            --capture-sizes 24 48 96 144 192 240 --warmup-rounds 1 --warmup-tokens 96 \
            --warmup-steps 8 --steady-cycles 10 --profile-start-step 8 --profile-steps 3 \
            > "$out/${backend}_launch.log" 2>&1
    done
done
