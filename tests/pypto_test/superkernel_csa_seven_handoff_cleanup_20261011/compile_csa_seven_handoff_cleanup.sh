#!/usr/bin/env bash
set -eo pipefail
private_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$private_root/vllm-ascend"
source "$private_root/env.sh"
export VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0
export PYPTO_BUILD_JOBS=2
output="$private_root/artifacts/csa_seven_handoff_cleanup_20261011"
mkdir -p "$output"
python - "$output" <<'PY'
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "tests/pypto_test")
from tmr_csa_full_128k_b16_compile import PACKAGES, compile_variant, shape_only_args

output = Path(sys.argv[1])
shapes = [(131072, b) for b in (4, 8, 16, 24)] + [(8192, b) for b in (16, 24, 32)]
checks = {}
for name in ("baseline", "all"):
    module = importlib.import_module(f"vllm_ascend.ops.pypto.{PACKAGES[name]}.decode_csa")
    module.decode_csa_tp1_layer_test._get_dep_graph()
    reference = None
    checks[name] = []
    for history, batch in shapes:
        arguments, _ = shape_only_args(module._decode_csa_tp1_layer, batch, history)
        program = str(module.decode_csa_tp1_layer_test.specialize(*arguments))
        if reference is None:
            reference = program
        assert program == reference, f"{name}: {history}/{batch} needs a separate compile"
        checks[name].append({"history": history, "batch": batch, "same_dynamic_program": True})
    compile_variant(name, output / "compiled" / name)
    print(f"STAGE: {name} compiled; all seven shapes use the same dynamic program", flush=True)
(output / "dynamic_shape_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
PY
