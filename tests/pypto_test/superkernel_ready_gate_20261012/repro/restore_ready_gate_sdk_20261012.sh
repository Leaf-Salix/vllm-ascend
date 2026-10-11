#!/usr/bin/env bash
set -eo pipefail
p="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$p/simpler"
source "$p/env.sh"
cmake -S . -B build/cp310-cp310-linux_aarch64 > "$p/artifacts/restore_ready_gate_binding_configure_20261012.log" 2>&1
cmake --build build/cp310-cp310-linux_aarch64 --target _task_interface --parallel "$PYPTO_BUILD_JOBS" \
  > "$p/artifacts/restore_ready_gate_binding_build_20261012.log" 2>&1
cmake --install build/cp310-cp310-linux_aarch64/python/bindings --prefix "$p/.venv/lib/python3.10/site-packages" \
  > "$p/artifacts/restore_ready_gate_binding_install_20261012.log" 2>&1
python - <<'PY' > "$p/artifacts/restore_ready_gate_runtime_build_20261012.log" 2>&1
from simpler_setup.runtime_builder import RuntimeBuilder
print(RuntimeBuilder(platform="a2a3").get_binaries("tensormap_and_ringbuffer", build=True))
PY
cd "$p/pypto"
cmake -S . -B build > "$p/artifacts/restore_ready_gate_torch_configure_20261012.log" 2>&1
cmake --build build --target pypto_core _torch_npu --parallel "$PYPTO_BUILD_JOBS" \
  > "$p/artifacts/restore_ready_gate_torch_build_20261012.log" 2>&1
cmake --install build --prefix "$p/.venv/lib/python3.10/site-packages/pypto" \
  > "$p/artifacts/restore_ready_gate_torch_install_20261012.log" 2>&1
python - <<'PY'
import _task_interface
from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
from pypto.torch.launch import _load_native
assert _load_native().simpler_revision == _task_interface.__build_commit__ == SIMPLER_KERNEL_REVISION
print("SDK revisions agree:", SIMPLER_KERNEL_REVISION, flush=True)
PY
