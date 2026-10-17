# SPDX-License-Identifier: Apache-2.0
"""CPU 编译核身份特化诊断，不启动 NPU。"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from core_specialization import specialize_core_identity


def main():
    from pypto import passes
    from pypto.runtime import CompileOptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((args.case / "entry.bindings.json").read_text())["_entry_contract"]
    preparation = json.loads((args.case / "preparation.json").read_text())
    if not preparation.get("compiled_control"):
        raise ValueError("必须先把同规格调度表精确编入控制条件")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "csa_static_spmd_feasibility_20261015"))
    spec = importlib.util.spec_from_file_location("csa_core_variant", args.case / "entry_jit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with specialize_core_identity(contract["leaf_tensor_count"], args.output / "core_specialization"):
        artifact = module.entry._resolve_kernel_artifact((), {"config": CompileOptions(
            platform="a2a3", output_dir=str(args.output / "kernel"),
            memory_planner=passes.MemoryPlanner.PYPTO)}, allow_signature_mode=True)
    result = {"source": str(args.case), "artifact": str(artifact.output_dir),
              "full_codegen_passed": True, "device_execution": False,
              "eligible_for_700us_target": False}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
