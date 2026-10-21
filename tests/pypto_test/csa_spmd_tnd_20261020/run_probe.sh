#!/usr/bin/env bash
# 单卡TND诊断；所有环境选择仅作用于本任务，复用已构建的固定工具链。
set -eo pipefail
: "${TASK_DEVICE:?必须通过task-submit --device auto分配设备}"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
[[ -f "$workspace/env-dsv4-0251rc1.sh" ]]
out=${1:?指定新结果目录}
shift
mode=probe
bsh_mode=false
if [[ ${1:-} == --performance ]]; then
    mode=performance
    shift
elif [[ ${1:-} == --bsh-probe ]]; then
    bsh_mode=true
    shift
elif [[ ${1:-} == --bsh-phase-profile ]]; then
    mode=phase_profile
    bsh_mode=true
    shift
fi
native_toolchain=false
if [[ ${1:-} == --native-toolchain ]]; then
    native_toolchain=true
    shift
fi
candidate_source="$repo/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd"
if [[ ${1:-} == --candidate-source ]]; then
    candidate_source=${2:?candidate source directory}
    shift 2
fi
baseline_source="$repo/vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
pypto_root="$workspace/.worktrees/pypto-csa-repro-20261019"
ptoas_root="$workspace/.worktrees/ptoas-csa-repro-20261019/build"
if [[ $bsh_mode == true ]]; then
    manifest="$repo/tests/pypto_test/csa_spmd_tnd_20261020/bsh_baseline_a241fa61_short_typefix_manifest.json"
    readarray -t bsh_roots < <(python - "$manifest" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for name in ("pypto", "ptoas"):
    root = Path(data[name]["source"])
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != data[name]["commit"]:
        raise RuntimeError(f"Frozen {name} source revision changed")
    for relative, expected in data[name]["copied_binaries"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen {name} binary changed: {relative}")
print(data["pypto"]["source"])
print(str(Path(data["ptoas"]["source"]) / "build"))
print(str(Path(data["bsh"]["snapshot"]) / "deepseek_v4_flash_csa_spmd"))
PY
)
    [[ ${#bsh_roots[@]} == 3 ]]
    pypto_root=${bsh_roots[0]}
    ptoas_root=${bsh_roots[1]}
    baseline_source=${bsh_roots[2]}
fi
if [[ $native_toolchain == true ]]; then
    pypto_root="$workspace/.worktrees/pypto-csa-tnd-perf-20261021"
    ptoas_root="$workspace/.worktrees/ptoas-csa-tnd-perf-20261021/build"
    [[ -f "$repo/tests/pypto_test/csa_spmd_tnd_20261020/native_toolchain_manifest.json" ]]
    [[ -x "$ptoas_root/bin/ptoas" ]]
fi
if [[ -e "$out" ]]; then
    printf '拒绝覆盖结果目录：%s\n' "$out" >&2
    exit 2
fi
source "$workspace/env-dsv4-0251rc1.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
if [[ $mode == performance || $mode == phase_profile ]]; then
    export PTO_CSA_VARIANT=performance
else
    export PTO_CSA_VARIANT=spmd_tnd
fi
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1800 HCCL_DETERMINISTIC=false
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 VLLM_BATCH_INVARIANT=0 LOCAL_WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True DYNAMIC_EPLB=false EXPERT_MAP_RECORD=false
export PYPTO_PROG_BUILD_DIR="$out/build_output" PYPTO_CACHE_DIR="$out/cache"
export ASCEND_PROCESS_LOG_PATH="${out}_device_logs" ASCEND_CACHE_PATH="${out}_ascend_cache"
export VLLM_CACHE_ROOT="${out}_vllm_cache"
mkdir -p "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_CACHE_PATH" "$VLLM_CACHE_ROOT"
helpers="$repo/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
export PYTHONPATH="$helpers:$repo/tests/pypto_test:${PYTHONPATH:-}"
unset HC_PRE_TEST_OPERATOR_SOURCE
snapshot="${out}_source"
python - "$repo" "$snapshot" "${BASH_SOURCE[0]}" "$mode" "$candidate_source" "$baseline_source" "${manifest:-}" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

repo, snapshot, launcher = map(Path, sys.argv[1:4])
mode = sys.argv[4]
snapshot.mkdir(parents=True, exist_ok=False)
shutil.copyfile(launcher, snapshot / "launcher.sh")
package = Path(sys.argv[5]).resolve()
shutil.copytree(package, snapshot / "spmd", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
baseline = Path(sys.argv[6]).resolve()
shutil.copytree(baseline, snapshot / "original_spmd",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
if sys.argv[7]:
    shutil.copyfile(sys.argv[7], snapshot / "bsh_baseline_manifest.json")
for path in (
    repo / "tests/pypto_test/csa_spmd_tnd_20261020/run_probe.py",
    repo / "tests/pypto_test/csa_spmd_tnd_20261020/service_probe.py",
    repo / "tests/pypto_test/csa_spmd_tnd_20261020/service_metadata_replay.py",
    repo / "tests/pypto_test/csa_spmd_tnd_20261020/tnd_performance.py",
    repo / "tests/pypto_test/csa_spmd_tnd_20261020/bsh_probe.py",
    repo / "tests/pypto_test/csa_spmd_tnd_20261020/metadata_matrix.py",
    repo / "tests/pypto_test/dsv4_csa_single_layer.py",
):
    shutil.copyfile(path, snapshot / path.name)
if mode in ("performance", "phase_profile"):
    helpers = repo / "tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
    target = snapshot / "perf_helpers"
    target.mkdir()
    for path in (*helpers.glob("*.py"), helpers / "source_interface.json"):
        shutil.copyfile(path, target / path.name)
    shutil.copyfile(snapshot / "dsv4_csa_single_layer.py", target / "dsv4_csa_single_layer.py")
manifest = {
    str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in snapshot.rglob("*") if path.is_file()
}
(snapshot / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
if [[ $native_toolchain == true ]]; then
    cp "$repo/tests/pypto_test/csa_spmd_tnd_20261020/native_toolchain_manifest.json" \
        "$snapshot/native_toolchain_manifest.json"
fi
if [[ $mode == performance ]]; then
    target="$snapshot/perf_helpers/run_device.py"
    source_args=(--spmd-source "$snapshot/spmd" --source-matrix-reference "$snapshot/original_spmd")
elif [[ $mode == phase_profile ]]; then
    target="$snapshot/bsh_probe.py"
    source_args=(--probe-script "$snapshot/perf_helpers/run_device.py"
        --wait-policy-source "$snapshot/original_spmd/wait_codegen.py"
        --spmd-source "$snapshot/spmd")
elif [[ $bsh_mode == true ]]; then
    target="$snapshot/bsh_probe.py"
    source_args=(--probe-script "$snapshot/run_probe.py"
        --wait-policy-source "$snapshot/original_spmd/wait_codegen.py"
        --spmd-source "$snapshot/spmd" --baseline-spmd-source "$snapshot/original_spmd")
    if [[ $native_toolchain == true ]]; then
        source_args+=(--toolchain-manifest "$snapshot/native_toolchain_manifest.json")
    fi
else
    target="$snapshot/run_probe.py"
    source_args=(--spmd-source "$snapshot/spmd")
fi
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    --pypto-root "$pypto_root" \
    --simpler-root "$workspace/.worktrees/simpler-csa-core-args-256-20261016" \
    --ptoas-root "$ptoas_root" \
    "$target" --output "$out" "${source_args[@]}" "$@"
