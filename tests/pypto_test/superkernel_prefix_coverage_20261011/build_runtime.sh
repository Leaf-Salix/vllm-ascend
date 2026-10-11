#!/usr/bin/env bash
set -eo pipefail
private_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$private_root/simpler"
source "$private_root/env.sh"
cmake -S . -B build/cp310-cp310-linux_aarch64 > "$private_root/artifacts/prefix_coverage_binding_configure.log" 2>&1
cmake --build build/cp310-cp310-linux_aarch64 --target _task_interface --parallel "$PYPTO_BUILD_JOBS" > "$private_root/artifacts/prefix_coverage_binding_build.log" 2>&1
cmake --install build/cp310-cp310-linux_aarch64/python/bindings --prefix "$private_root/.venv/lib/python3.10/site-packages" > "$private_root/artifacts/prefix_coverage_binding_install.log" 2>&1
python - <<'PY' > "$private_root/artifacts/prefix_coverage_runtime_build.log" 2>&1
from simpler_setup.runtime_builder import RuntimeBuilder
print(RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=True))
PY
printf '%s\n' 'STAGE: private Simpler binding and TMR runtime built'
cd "$private_root/pypto"
cmake -S . -B build > "$private_root/artifacts/prefix_coverage_torch_configure.log" 2>&1
cmake --build build --target _torch_npu --parallel "$PYPTO_BUILD_JOBS" > "$private_root/artifacts/prefix_coverage_torch_build.log" 2>&1
cmake --install build --prefix "$private_root/.venv/lib/python3.10/site-packages/pypto" > "$private_root/artifacts/prefix_coverage_torch_install.log" 2>&1
python - <<'PY'
import _task_interface
from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
from pypto.torch.launch import _load_native
assert _load_native().simpler_revision == _task_interface.__build_commit__ == SIMPLER_KERNEL_REVISION
print("STAGE: three SDK revisions match", SIMPLER_KERNEL_REVISION, flush=True)
PY
