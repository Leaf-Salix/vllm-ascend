#!/usr/bin/env bash
# 当前CSA TND的16卡功能验证，源码冻结、SDK共享、KV只读。
set -eo pipefail
: "${TASK_DEVICE:?必须经task-submit分配16卡}"
source_tree=$(realpath "${1:?冻结源码}")
output=$(realpath -m "${2:?输出目录}")
variant=${3:?native或spmd_tnd或mpmd_tnd}
shift 3
batch=16
budget=0.92
single_card_gate=
ring_heap_mb=
ring_task_window=
hccl_buffer_mb=1800
if [[ $# -gt 0 && "$1" != --* ]]; then batch=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then budget=$1; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then single_card_gate=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ring-heap-mb) ring_heap_mb=${2:?}; shift 2 ;;
        --ring-task-window) ring_task_window=${2:?}; shift 2 ;;
        --hccl-buffer-mb) hccl_buffer_mb=${2:?}; shift 2 ;;
        --device) shift 2 ;; # 队列已通过TASK_DEVICE指定实际设备
        *) exit 2 ;;
    esac
done
if [[ "$variant" != native && "$variant" != spmd_tnd && "$variant" != mpmd_tnd ]]; then exit 2; fi
if [[ "$batch" != 16 && "$batch" != 24 ]]; then exit 2; fi
record=$(dirname "$source_tree")/manifest.json
repo=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$record")
workspace=$(dirname "$repo")
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
bank="$workspace/csa-repro-20261021-clean/kv-repro/mixed128k_fixed"
mkdir "$output"
python - "$record" "$CSA_SHARED_TOOLCHAIN_MANIFEST" "$bank" "$output" "$variant" "$batch" "$budget" "$single_card_gate" "$ring_heap_mb" "$ring_task_window" "$hccl_buffer_mb" <<'PY'
import json, pathlib, sys
record, shared, bank, output, variant, batch, budget, single_card_gate, heap, window, hccl_buffer = sys.argv[1:]
assert int(hccl_buffer) > 0
assert bool(heap) == bool(window), 'ring配置须完整提供'
ring_config = {'heap_mb': [int(x) for x in heap.split(',')], 'task_window': int(window)} if heap else None
if ring_config:
    assert len(ring_config['heap_mb']) == 4 and all(x > 0 for x in ring_config['heap_mb'])
    assert ring_config['task_window'] >= 4 and not (ring_config['task_window'] & (ring_config['task_window'] - 1))
data = json.load(open(record))
assert data['toolchain'] == json.load(open(shared)), '公共SDK变化，请重新冻结'
assert data['toolchain']['status'] == 'validated'
if data.get('source_overlays') and variant != 'native':
    assert single_card_gate, '工作区修复必须先通过单卡验证'
    gate = json.load(open(single_card_gate))
    assert gate['status'] == 'PASS', '单卡验证尚未通过，禁止启动整模型'
    assert set(gate['variants']) == {'deepseek_v4_flash_csa_spmd_tnd', 'deepseek_v4_flash_csa_tnd'}
    assert all(row['status'] == 'PASS' and row['guard_status'] == 'PASS' for row in gate['variants'].values())
    if ring_config:
        assert gate.get('batch') == int(batch) and gate.get('ring_config') == ring_config, 'ring配置须有同档单卡证据'
    data['single_card_gate'] = single_card_gate
plan = json.load(open(pathlib.Path(bank) / 'plan.json'))
assert plan['mixed_requests'] and len(plan['cases']) >= int(batch)
data.update(variant=variant, batch=int(batch), gpu_memory_utilization=float(budget), bank=bank,
            selected_cases=plan['cases'][:int(batch)], decode_tokens=192,
            TP=1, DP=16, EP=16, HCA='native', NZ=2, EPLB=False, static_kernel=False, superkernel=False,
            graph_mode='FULL_DECODE_ONLY', deterministic_level=1, cann='9.2.0-beta.2',
            pto_ring_config=ring_config, hccl_buffer_mb=int(hccl_buffer))
(pathlib.Path(output) / 'configuration.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
PY
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_tree:$source_tree/tests/pypto_test:$workspace/.cache/migration-v0.25.1rc1/vllm:$PYTHONPATH"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export PTO_CSA_VARIANT="$variant"
if [[ -n "$ring_heap_mb" ]]; then
    export PTO_CSA_RING_HEAP_MB="$ring_heap_mb" PTO_CSA_RING_TASK_WINDOW="$ring_task_window"
else
    unset PTO_CSA_RING_HEAP_MB PTO_CSA_RING_TASK_WINDOW
fi
[[ "$variant" == native ]] && export PTO_CSA_VARIANT=spmd_tnd
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE="$hccl_buffer_mb" HCCL_DETERMINISTIC=true
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 LOCAL_WORLD_SIZE=1
export DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export ASCEND_WORK_PATH="$output/ascend" ASCEND_PROCESS_LOG_PATH="$output/ascend/log"
export PYPTO_PROG_BUILD_DIR="$output/build" PYPTO_CACHE_DIR="$output/pypto_cache"
export VLLM_CACHE_ROOT="$output/vllm_cache" ASCEND_CACHE_PATH="$output/ascend_cache"
export TORCH_EXTENSIONS_DIR="$workspace/csa-repro-20261021-clean/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$output/inductor_cache" TRITON_CACHE_DIR="$output/triton_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$PYPTO_CACHE_DIR" "$VLLM_CACHE_ROOT" \
    "$PYPTO_PROG_BUILD_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$output/compile"
export ASCEND_OPP_PATH=$(python "$source_tree/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$output/private_opp")
backend=pto
[[ "$variant" == native ]] && backend=native
cd "$output/compile"
python "$source_tree/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --shared-toolchain --workspace-root "$workspace" --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$source_tree/tests/pypto_test/offline_pd/run.py" decode \
    --bank "$bank" --output "$output" --backend "$backend" --pto-attention csa --mixed-requests \
    --decode-dp 16 --gpu-memory-utilization "$budget" --batch "$batch" --deterministic \
    --decode-tokens 192 --weight-nz-mode 2 --graph-mode full_decode_only --no-static-kernel \
    --capture-sizes 6 "$((batch * 6))" --event-work-mode 1 --port 30881 \
    > "$output/launch.log" 2>&1
python "$source_tree/tests/pypto_test/low_acceptance_20261001/mixed.py" \
    --root "$output" --bank "$bank" --batch "$batch" --mean-range 1 5 > "$output/acceptance_summary.log"
python "$source_tree/tests/pypto_test/csa_tnd_model_validation_20261023/check.py" "$output"
