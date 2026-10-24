#!/usr/bin/env bash
# HCA整模型功能验证：沿用CSA已验证的PD bank与DP/EP16配置，不测性能。
set -eo pipefail
: "${TASK_DEVICE:?必须经task-submit分配16卡}"
hca_source=$(realpath "${1:?冻结源码}")
hca_output=$(realpath -m "${2:?新的输出目录}")
hca_side=${3:?native或pto}
hca_batch=${4:?16或24}
hca_gate=$(realpath "${5:?单卡结果}")
shift 5
while [[ $# -gt 0 ]]; do
    case "$1" in --device) shift 2 ;; *) exit 2 ;; esac
done
[[ "$hca_side" == native || "$hca_side" == pto ]]
[[ "$hca_batch" == 16 || "$hca_batch" == 24 ]]
hca_record=$(dirname "$hca_source")/manifest.json
hca_repo=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$hca_record")
hca_workspace=$(dirname "$hca_repo")
source "$hca_workspace/env-dsv4-0251rc1.sh"
source "$hca_repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$hca_repo/tests/pypto_test/csa_shared_toolchain/env.sh"
hca_bank="$hca_workspace/csa-repro-20261021-clean/kv-repro/mixed128k_fixed"
hca_budget=0.92
hca_hccl=1800
if [[ "$hca_batch" == 24 ]]; then hca_budget=0.935; hca_hccl=768; fi
mkdir "$hca_output"
python - "$hca_record" "$CSA_SHARED_TOOLCHAIN_MANIFEST" "$hca_gate" "$hca_bank" "$hca_output" "$hca_side" "$hca_batch" "$hca_budget" "$hca_hccl" <<'PY'
import json,pathlib,sys
record,shared,gate,bank,output,side,batch,budget,hccl=sys.argv[1:]
data=json.load(open(record)); current=json.load(open(shared)); validation=json.load(open(gate))
assert data['toolchain']==current and current['status']=='validated'
assert validation['status']=='PASS' and validation['framework_graph_replay'] and validation['service_binding']
assert validation['toolchain']==current
assert {s['case'] for s in validation['scenes']}=={'C04','C08','C05','S1','S5','B1'}
assert validation['ring_config']==dict(heap_mb=[64,256,64,64],task_window=4096)
assert all(step['python_not_resubmitted'] and step['graph_contract_allows'] for s in validation['scenes'] for step in s['steps'])
plan=json.load(open(pathlib.Path(bank)/'plan.json'));assert plan['mixed_requests']
data.update(backend=side,variant='hca_mpmd_tnd' if side=='pto' else 'native',
            batch=int(batch),decode_tokens=192,gpu_memory_utilization=float(budget),bank=bank,
            selected_cases=plan['cases'][:int(batch)],single_card_gate=gate,
            TP=1,DP=16,EP=16,HCA='mpmd_tnd' if side=='pto' else 'native',CSA='native',NZ=2,EPLB=False,
            static_kernel=False,superkernel=False,graph_mode='FULL_DECODE_ONLY',deterministic_level=1,
            cann='9.2.0-beta.2',pto_ring_config=validation['ring_config'],hccl_buffer_mb=int(hccl),
            scope='沿用CSA功能配置，要求真实图重放；不是整模型性能对比')
(pathlib.Path(output)/'configuration.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
PY
export LD_LIBRARY_PATH="$hca_repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$hca_source:$hca_source/tests/pypto_test:$hca_workspace/.cache/migration-v0.25.1rc1/vllm:$PYTHONPATH"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export PTO_CSA_RING_HEAP_MB=64,256,64,64 PTO_CSA_RING_TASK_WINDOW=4096
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE="$hca_hccl" HCCL_DETERMINISTIC=true
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 LOCAL_WORLD_SIZE=1 DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export ASCEND_WORK_PATH="$hca_output/ascend" ASCEND_PROCESS_LOG_PATH="$hca_output/ascend/log"
export PYPTO_PROG_BUILD_DIR="$hca_output/build" PYPTO_CACHE_DIR="$hca_output/pypto_cache"
export VLLM_CACHE_ROOT="$hca_output/vllm_cache" ASCEND_CACHE_PATH="$hca_output/ascend_cache"
export TORCH_EXTENSIONS_DIR="$hca_workspace/csa-repro-20261021-clean/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$hca_output/inductor_cache" TRITON_CACHE_DIR="$hca_output/triton_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$PYPTO_CACHE_DIR" "$VLLM_CACHE_ROOT" \
    "$PYPTO_PROG_BUILD_DIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$hca_output/compile"
ASCEND_OPP_PATH=$(python "$hca_source/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$hca_output/private_opp")
export ASCEND_OPP_PATH
cd "$hca_output/compile"
python "$hca_source/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$hca_workspace" --vllm-root "$hca_workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$hca_source/tests/pypto_test/offline_pd/run.py" decode \
    --bank "$hca_bank" --output "$hca_output" --backend "$hca_side" --pto-attention hca --mixed-requests \
    --decode-dp 16 --gpu-memory-utilization "$hca_budget" --batch "$hca_batch" --deterministic \
    --hccl-buffer-mb "$hca_hccl" --decode-tokens 192 --weight-nz-mode 2 \
    --graph-mode full_decode_only --no-static-kernel --capture-sizes 6 "$((hca_batch * 6))" \
    --event-work-mode 1 --port 30991 > "$hca_output/launch.log" 2>&1
python "$hca_source/tests/pypto_test/low_acceptance_20261001/mixed.py" \
    --root "$hca_output" --bank "$hca_bank" --batch "$hca_batch" --mean-range 1 5 \
    > "$hca_output/acceptance_summary.log"
python "$hca_repo/tests/pypto_test/hca_tnd_model_validation_20261024/check_model.py" "$hca_output"
