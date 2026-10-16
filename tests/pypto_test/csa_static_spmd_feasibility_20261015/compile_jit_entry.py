# SPDX-License-Identifier: Apache-2.0
"""从 JIT 源码在 CPU 编译 program artifact；不启动 kernel runtime。"""

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

from pypto import passes
from pypto.runtime import CompileOptions, RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-planner", choices=("PYPTO", "DSA_RP", "PTOAS"), default="PTOAS")
    parser.add_argument("--kernel-mode", action="store_true", help="只生成 native 直调的 kernel artifact，不创建 Worker")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"source": str(args.source), "frontend": "jit", "artifact_mode": "kernel" if args.kernel_mode else "program",
              "source_imported": False, "full_codegen_passed": False, "device_execution": False,
              "kernel_runtime_validated": False, "runtime_abi_supported": False, "memory_planner": args.memory_planner}
    try:
        snapshot = args.output / "source.py"
        snapshot.write_text(args.source.read_text())
        result["source_snapshot"] = str(snapshot.resolve())
        spec = importlib.util.spec_from_file_location("csa_static_prepared_jit", snapshot)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result["source_imported"] = True
        if args.kernel_mode:
            compiled = module.entry._resolve_kernel_artifact((), {"config": CompileOptions(
                platform="a2a3", output_dir=str(args.output / "kernel"), dump_passes=True,
                memory_planner=getattr(passes.MemoryPlanner, args.memory_planner))}, allow_signature_mode=True)
        else:
            compiled = module.entry.compile(config=RunConfig(
                platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "kernel"),
                dump_passes=True, memory_planner=getattr(passes.MemoryPlanner, args.memory_planner)))
        result.update(full_codegen_passed=True, output_dir=str(compiled.output_dir))
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
