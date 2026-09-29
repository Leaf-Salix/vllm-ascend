#!/usr/bin/env bash
set -eo pipefail
: "${TASK_DEVICE:?Submit through task-submit --device auto}"
workspace=/data/pyptouser/qinchuanyu/pto-eager
repo="$workspace/vllm-ascend-dsv4-pto-0251rc1"
root="$repo/tests/pypto_test/results/csa_hc_post_resident_20260929/boundary"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
export PTO_CSA_VARIANT=pkg:dsv4_csa_hc_post_resident_55b89ee2
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 VLLM_ASCEND_ENABLE_NZ=2
export PTO_CSA_RING_HEAP_MB=256,128,256,32 PTO_CSA_RING_TASK_WINDOW=4096
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_BATCH_INVARIANT=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=true
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
[[ "$ASCEND_HOME_PATH" == */cann-9.2.0-beta.2 ]]
for side in baseline candidate; do
    source_repo="$workspace/.cache/csa-hc-post-resident-55b89ee2-$side"
    out="$root/$side"
    test ! -e "$out/report.json"
    mkdir -p "$out/ascend" "$out/ascend_cache"
    export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
    export VLLM_CACHE_ROOT="$out/vllm_cache"
    cd "$out"
    python - "$TASK_DEVICE" "$source_repo" "$out/run_context.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[3]).write_text(json.dumps({
    "device": int(sys.argv[1]), "source": sys.argv[2],
    "cann": os.environ["ASCEND_HOME_PATH"], "variant": os.environ["PTO_CSA_VARIANT"],
}, indent=2) + "\n")
PY
    # T18 leaves two valid tokens in the final four-token HC_post worker.
    # Reuse the same graph with active B=3/2/1/3 to check padding and guards.
    python "$source_repo/tests/pypto_test/coefficients_seven_experiment/accuracy_case.py" "$source_repo" \
        --checkpoint /data/model/DeepSeek-V4-Flash-0731-w8a8 \
        --output "$out" --device "$TASK_DEVICE" --batch 3 --history 127 \
        --layer-index 4 --variant "$PTO_CSA_VARIANT" --weight-nz-mode 2 --seed 1024 \
        --atomic-add 0 --deterministic-level 1 --save-state --padding-graph > "$out/run.log" 2>&1
done
