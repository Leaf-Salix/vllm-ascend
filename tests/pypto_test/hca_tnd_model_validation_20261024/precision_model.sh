#!/usr/bin/env bash
# 真实B16首token诊断；两侧相同模型壳与容量，均不改采样或数学实现。
set -eo pipefail
: "${TASK_DEVICE:?需经task-submit分配16卡}"
precision_source=$(realpath "${1:?冻结源码}")
precision_output=$(realpath -m "${2:?新的结果目录}")
precision_mode=${3:?native或pto}
shift 3
while [[ $# -gt 0 ]]; do
    case "$1" in --device) shift 2 ;; *) exit 2 ;; esac
done
[[ "$precision_mode" == native || "$precision_mode" == pto ]]
precision_record=$(dirname "$precision_source")/manifest.json
precision_repo=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$precision_record")
precision_workspace=$(dirname "$precision_repo")
source "$precision_workspace/env-dsv4-0251rc1.sh"
source "$precision_repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$precision_repo/tests/pypto_test/csa_shared_toolchain/env.sh"
precision_bank="$precision_workspace/csa-repro-20261021-clean/kv-repro/mixed128k_fixed"
mkdir "$precision_output"
python - "$precision_record" "$CSA_SHARED_TOOLCHAIN_MANIFEST" "$precision_output" "$precision_mode" "$precision_bank" <<'PY'
import json,pathlib,sys
record,shared,output,mode,bank=sys.argv[1:]
data=json.load(open(record));assert data['toolchain']==json.load(open(shared));assert data['toolchain']['status']=='validated'
data.update(mode=mode,bank=bank,batch=16,TP=1,DP=16,EP=16,decode_tokens=1,capacity_decode_tokens=192,
            HCA=mode,CSA='native',gpu_memory_utilization=0.92,HCCL_BUFFSIZE=1800,NZ=2,deterministic_level=1,
            ring_config=dict(heap_mb=[64,256,64,64],task_window=4096),performance_measured=False)
(pathlib.Path(output)/'configuration.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
PY
export HCA_PRECISION_MODE="$precision_mode" HCA_PRECISION_OUTPUT="$precision_output/precision"
export LD_LIBRARY_PATH="$precision_repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$precision_source:$precision_source/tests/pypto_test:$precision_workspace/.cache/migration-v0.25.1rc1/vllm:$PYTHONPATH"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export PTO_CSA_RING_HEAP_MB=64,256,64,64 PTO_CSA_RING_TASK_WINDOW=4096
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=true
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 LOCAL_WORLD_SIZE=1 DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export ASCEND_WORK_PATH="$precision_output/ascend" ASCEND_PROCESS_LOG_PATH="$precision_output/ascend/log"
export PYPTO_PROG_BUILD_DIR="$precision_output/build" PYPTO_CACHE_DIR="$precision_output/pypto_cache"
export VLLM_CACHE_ROOT="$precision_output/vllm_cache" ASCEND_CACHE_PATH="$precision_output/ascend_cache"
export TORCH_EXTENSIONS_DIR="$precision_workspace/csa-repro-20261021-clean/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$precision_output/inductor_cache" TRITON_CACHE_DIR="$precision_output/triton_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$PYPTO_CACHE_DIR" "$VLLM_CACHE_ROOT" \
    "$PYPTO_PROG_BUILD_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$precision_output/compile"
ASCEND_OPP_PATH=$(python "$precision_source/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$precision_output/private_opp")
export ASCEND_OPP_PATH
cd "$precision_output/compile"
python "$precision_source/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$precision_workspace" --vllm-root "$precision_workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$precision_source/tests/pypto_test/offline_pd/run.py" decode \
    --bank "$precision_bank" --output "$precision_output" --backend "$precision_mode" --pto-attention hca --mixed-requests \
    --decode-dp 16 --gpu-memory-utilization 0.92 --batch 16 --deterministic --hccl-buffer-mb 1800 \
    --decode-tokens 1 --weight-nz-mode 2 --graph-mode full_decode_only --no-static-kernel \
    --capture-sizes 6 96 --event-work-mode 1 --port 31091 > "$precision_output/launch.log" 2>&1
