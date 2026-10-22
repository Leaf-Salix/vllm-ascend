#!/usr/bin/env bash
# 真实TP1/DP=EP16混合接受率验证，必须由task-submit分配16卡。
set -eo pipefail
: "${TASK_DEVICE:?请通过task-submit分配16卡}"
source_tree="$(realpath "${1:?冻结源码}")"
output="$(realpath -m "${2:?新输出目录}")"
variant="${3:?spmd_tnd或mpmd_tnd}"
if [[ "$variant" != spmd_tnd && "$variant" != mpmd_tnd ]]; then exit 2; fi
record="$(dirname "$source_tree")/manifest.json"
repo="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["repository"])' "$record")"
workspace="$(dirname "$repo")"
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/results/csa_native_template_20260929/env.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
bank="$repo/tests/pypto_test/results/low_acceptance_20261001/mixed8k_fixed"
mkdir "$output"
python - "$record" "$CSA_SHARED_TOOLCHAIN_MANIFEST" "$bank" "$output" "$variant" <<'PY'
import json, pathlib, sys
record, shared, bank, out, variant = sys.argv[1:]
manifest = json.load(open(record))
assert manifest['toolchain'] == json.load(open(shared)), '公共SDK变化，停止跨版本比较'
plan = json.load(open(pathlib.Path(bank) / 'plan.json'))
assert len(plan['cases']) >= 16 and plan['mixed_requests']
selected = plan['cases'][:16]
assert len({c['key'] for c in selected}) == 16
manifest.update(variant=variant, bank=bank, batch=16, decode_tokens=192, selected_cases=selected,
                TP=1, DP=16, EP=16, HCA='native', NZ=2, EPLB=False,
                scope='真实整模型正确性；不代替八档L1性能验收')
(pathlib.Path(out) / 'configuration.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
PY
mkdir "$output/compile"
cd "$output/compile"
export LD_LIBRARY_PATH="$repo/.cache/csa/native-install:$LD_LIBRARY_PATH"
export PYTHONPATH="$source_tree:$source_tree/tests/pypto_test:$workspace/.cache/migration-v0.25.1rc1/vllm:$PYTHONPATH"
export VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_RUNTIME=tensormap_and_ringbuffer PTO_CSA_VARIANT="$variant"
export HCCL_DETERMINISTIC=true DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export ASCEND_PROCESS_LOG_PATH="$output/ascend" PYPTO_PROG_BUILD_DIR="$output/build"
export PYPTO_CACHE_DIR="$output/pypto_cache" VLLM_CACHE_ROOT="$output/vllm_cache" ASCEND_CACHE_PATH="$output/ascend_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$PYPTO_CACHE_DIR" "$VLLM_CACHE_ROOT" "$PYPTO_PROG_BUILD_DIR"
export ASCEND_OPP_PATH="$(python "$source_tree/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$output/private_opp")"
python "$source_tree/tests/pypto_test/offline_pd/run.py" decode \
    --bank "$bank" --output "$output" --backend pto --pto-attention csa --mixed-requests \
    --decode-dp 16 --gpu-memory-utilization 0.95 --batch 16 --deterministic \
    --decode-tokens 192 --max-num-batched-tokens 256 --weight-nz-mode 2 \
    --graph-mode full_decode_only --capture-sizes 6 96 --port 30871 \
    > "$output/launch.log" 2>&1
python "$source_tree/tests/pypto_test/low_acceptance_20261001/mixed.py" \
    --root "$output" --bank "$bank" --batch 16 --mean-range 1 5 > "$output/acceptance_summary.log"
