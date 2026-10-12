#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
output="$p/artifacts/boundary_compact_sdk_20261012"
mkdir -p "$output"
cd "$p/simpler"
source "$p/env.sh"
export PYPTO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 PYPTO_CODEGEN_MAX_WORKERS=1
cmake -S . -B build/cp310-cp310-linux_aarch64 > "$output/binding_configure.log" 2>&1
cmake --build build/cp310-cp310-linux_aarch64 --target _task_interface --parallel "$PYPTO_BUILD_JOBS" \
    > "$output/binding_build.log" 2>&1
cmake --install build/cp310-cp310-linux_aarch64/python/bindings --prefix "$p/.venv/lib/python3.10/site-packages" \
    > "$output/binding_install.log" 2>&1
python - <<'INNER' > "$output/runtime_build.log" 2>&1
from simpler_setup.runtime_builder import RuntimeBuilder
print(RuntimeBuilder(platform='a2a3').get_binaries('tensormap_and_ringbuffer', build=True))
INNER
cd "$p/pypto"
cmake -S . -B build > "$output/torch_configure.log" 2>&1
cmake --build build --target pypto_core _torch_npu --parallel "$PYPTO_BUILD_JOBS" \
    > "$output/torch_build.log" 2>&1
cmake --install build --prefix "$p/.venv/lib/python3.10/site-packages/pypto" \
    > "$output/torch_install.log" 2>&1
python - <<'INNER' > "$output/sdk_revision.log" 2>&1
import _task_interface
from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
from pypto.torch.launch import _load_native
assert _load_native().simpler_revision == _task_interface.__build_commit__ == SIMPLER_KERNEL_REVISION
print('SDK revisions agree:', SIMPLER_KERNEL_REVISION)
INNER
python "$p/pypto/tests/st/runtime/kernel/task_boundary_acceptance.py" compile --compiled "$output/compiled" --modes plain early --scalar-steps 256 > "$output/compile.log" 2>&1
cat "$output/sdk_revision.log"
tail -2 "$output/compile.log"
