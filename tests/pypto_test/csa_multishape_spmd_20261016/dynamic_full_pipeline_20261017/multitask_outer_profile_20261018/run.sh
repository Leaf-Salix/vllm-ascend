#!/usr/bin/env bash
# 只补原多task的128K/B28外层完整设备跨度，必须由队列分配NPU。
set -eo pipefail
: "${TASK_DEVICE:?必须通过task-submit --device auto分配设备}"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(git -C "$here" rev-parse --show-toplevel)
workspace_dir=$(cd -- "$repo_dir/.." && pwd)
out=${1:?指定全新输出目录}
source "$workspace_dir/env-dsv4-0251rc1.sh"
export CSA_STATIC_PYPTO_ROOT="$workspace_dir/.worktrees/pypto-csa-memory-reuse-20261018"
export CSA_STATIC_PYPTO_IMPORT_MODE=filesystem
export HCA_SIMPLER_ROOT="$workspace_dir/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 PTO_CSA_VARIANT=performance
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
if [[ -e "$out" ]]; then
    printf '输出目录已存在，禁止覆盖: %s\n' "$out" >&2
    exit 2
fi
mkdir -p "$out/ascend" "$out/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$out/ascend" ASCEND_CACHE_PATH="$out/ascend_cache"
export VLLM_CACHE_ROOT="$out/vllm_cache"
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export HC_PRE_TEST_OPERATOR_SOURCE="$repo_dir/vllm_ascend/ops/pypto"
export ASCEND_OPP_PATH="$(python "$repo_dir/tests/pypto_test/dsv4_hca_prepare_opp.py" --destination "$out/opp_root")"
mkdir -p "$ASCEND_OPP_PATH/static_kernel" "$workspace_dir/.cache/a"
work_dir=$(mktemp -d "$workspace_dir/.cache/a/p28_XXXXXX")
printf '%s\n' "$work_dir" > "$out/compile_workdir.txt"
python - "$out" "$repo_dir" "$here" <<'PY'
import json, os, pathlib, shutil, subprocess, sys
out, repo, here = map(pathlib.Path, sys.argv[1:])
source = repo / 'vllm_ascend/ops/pypto/deepseek_v4_flash_csa'
saved = out / 'source_snapshot/deepseek_v4_flash_csa'
saved.mkdir(parents=True)
for path in source.glob('*.py'):
    shutil.copyfile(path, saved / path.name)
for name in ('csa_compiled_case.py', 'csa_measure.py', 'run.sh', 'summarize.py'):
    shutil.copyfile(here / name, out / name)
keys = ('TASK_DEVICE', 'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'CSA_STATIC_PYPTO_ROOT',
        'CSA_STATIC_PYPTO_IMPORT_MODE', 'HCA_SIMPLER_ROOT', 'PTOAS_ROOT',
        'VLLM_ASCEND_ENABLE_NZ', 'VLLM_ASCEND_PTO_CSA_ATOMIC_ADD', 'PTO_CSA_VARIANT')
audit = {'source': str(source), 'commit': subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'environment': {key: os.environ.get(key) for key in keys},
    'python': sys.executable, 'source_snapshot': str(saved),
    'scope': '原PTO分task完整服务CSA；外部设备跨度包含AICPU调度；不运行Native/静态/动态单SPMD'}
(out / 'environment.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
PY
cd -- "$work_dir"
python "$repo_dir/tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py" \
    "$here/csa_compiled_case.py" --side pto --history 131072 --batch 28 \
    --super-kernel 0 --device "$TASK_DEVICE" --output "$out" > "$out/run.log" 2>&1
python "$here/summarize.py" "$out"
