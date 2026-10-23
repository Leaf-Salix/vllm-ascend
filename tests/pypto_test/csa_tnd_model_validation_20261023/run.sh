#!/usr/bin/env bash
# 当前CSA TND的16卡功能验证，源码冻结、SDK共享、KV只读。
set -eo pipefail
: "${TASK_DEVICE:?必须经task-submit分配16卡}"
source_tree=$(realpath "${1:?冻结源码}")
output=$(realpath -m "${2:?输出目录}")
variant=${3:?native或spmd_tnd或mpmd_tnd}
batch=${4:-16}
budget=${5:-0.92}
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
python - "$record" "$CSA_SHARED_TOOLCHAIN_MANIFEST" "$bank" "$output" "$variant" "$batch" "$budget" <<'PY'
import json, pathlib, sys
record, shared, bank, output, variant, batch, budget = sys.argv[1:]
data = json.load(open(record))
assert data['toolchain'] == json.load(open(shared)), '公共SDK变化，请重新冻结'
assert data['toolchain']['status'] == 'validated'
plan = json.load(open(pathlib.Path(bank) / 'plan.json'))
assert plan['mixed_requests'] and len(plan['cases']) >= int(batch)
data.update(variant=variant, batch=int(batch), gpu_memory_utilization=float(budget), bank=bank,
            selected_cases=plan['cases'][:int(batch)], decode_tokens=192,
            TP=1, DP=16, EP=16, HCA='native', NZ=2, EPLB=False, static_kernel=False, superkernel=False,
            graph_mode='FULL_DECODE_ONLY', deterministic_level=1, cann='9.2.0-beta.2')
(pathlib.Path(output) / 'configuration.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
PY
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_tree:$source_tree/tests/pypto_test:$workspace/.cache/migration-v0.25.1rc1/vllm:$PYTHONPATH"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_RUNTIME=tensormap_and_ringbuffer
export PTO_CSA_VARIANT="$variant"
[[ "$variant" == native ]] && export PTO_CSA_VARIANT=spmd_tnd
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=true
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
