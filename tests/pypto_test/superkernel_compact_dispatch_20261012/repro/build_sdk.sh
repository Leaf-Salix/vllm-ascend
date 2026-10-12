#!/usr/bin/env bash
set -eo pipefail
p="/data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009"
source "$p/sdk-compact-dispatch-20261012/env.sh"
a="$p/artifacts/boundary_compact_dispatch_20261012"
s="$p/simpler-compact-dispatch-20261012"
q="$p/pypto-compact-dispatch-20261012"
cmake -S "$s" -B "$s/build/compact-sdk" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$COMPACT_SDK_ROOT/.venv/bin/python" -DCMAKE_PREFIX_PATH="$p/.venv/lib/python3.10/site-packages;/data/pyptouser/qinchuanyu/pto-eager/.venv-dsv4-0251rc1/lib/python3.10/site-packages" > "$a/binding_configure.log" 2>&1
cmake --build "$s/build/compact-sdk" --target _task_interface --parallel 8 > "$a/binding_build.log" 2>&1
cmake --install "$s/build/compact-sdk/python/bindings" --prefix "$COMPACT_SDK_ROOT/.venv/lib/python3.10/site-packages" > "$a/binding_install.log" 2>&1
cmake -S "$q" -B "$q/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython_EXECUTABLE="$COMPACT_SDK_ROOT/.venv/bin/python" -DPYPTO_BUILD_TORCH_NPU=ON -DCMAKE_PREFIX_PATH="$COMPACT_SDK_ROOT/.venv/lib/python3.10/site-packages;$p/.venv/lib/python3.10/site-packages;/data/pyptouser/qinchuanyu/pto-eager/.venv-dsv4-0251rc1/lib/python3.10/site-packages" > "$a/adapter_configure.log" 2>&1
cmake --build "$q/build" --target _torch_npu --parallel 8 > "$a/adapter_build.log" 2>&1
python - <<'INNER' > "$a/sdk_revision.log" 2>&1
import _task_interface
from pypto._kernel_abi import SIMPLER_KERNEL_REVISION
from pypto.torch.launch import _load_native
assert _load_native().simpler_revision == _task_interface.__build_commit__ == SIMPLER_KERNEL_REVISION
print('SDK revisions agree:', SIMPLER_KERNEL_REVISION)
INNER
python "$a/repro/driver.py" build --root "$a" > "$a/build.log" 2>&1
python "$a/repro/driver.py" compile --root "$a" --compiled "$a/compiled" --modes plain early --scalar-steps 256 > "$a/compile.log" 2>&1
