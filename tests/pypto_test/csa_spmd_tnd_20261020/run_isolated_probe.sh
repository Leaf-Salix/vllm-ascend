#!/usr/bin/env bash
# Correctness-only probe; every device execution must be assigned by task-submit.
set -eo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
workspace=$repo
while [[ ! -f "$workspace/env-dsv4-0251rc1.sh" && "$workspace" != / ]]; do
    workspace=$(dirname -- "$workspace")
done
probe_directory=${1:?frozen probe directory}
probe_output=${2:?fresh output directory}
shift 2
probe_cpu=false
for probe_argument in "$@"; do
    if [[ $probe_argument == --cpu-only ]]; then probe_cpu=true; fi
done
if [[ $probe_cpu == false ]]; then : "${TASK_DEVICE:?must use task-submit}"; fi
[[ ! -e "$probe_output" ]]
source "$workspace/env-dsv4-0251rc1.sh"
source "$repo/tests/pypto_test/csa_shared_toolchain/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 VLLM_ASCEND_ENABLE_NZ=2
export PYPTO_PROG_BUILD_DIR="${probe_output}_programs" PYPTO_CACHE_DIR="${probe_output}_cache"
export ASCEND_PROCESS_LOG_PATH="${probe_output}_device_logs" ASCEND_CACHE_PATH="${probe_output}_ascend_cache"
helpers="$repo/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017"
export PYTHONPATH="$probe_directory:$helpers:$repo/tests/pypto_test:${PYTHONPATH:-}"
exec python "$repo/tests/pypto_test/csa_toolchain_repro_20261019/bootstrap.py" \
    --workspace-root "$workspace" \
    --shared-toolchain \
    --vllm-root "$workspace/.cache/migration-v0.25.1rc1/vllm" \
    "$repo/tests/pypto_test/csa_spmd_tnd_20261020/run_isolated_probe.py" --probe-directory "$probe_directory" \
    --toolchain-manifest "$CSA_SHARED_TOOLCHAIN_MANIFEST" --output "$probe_output" "$@"
