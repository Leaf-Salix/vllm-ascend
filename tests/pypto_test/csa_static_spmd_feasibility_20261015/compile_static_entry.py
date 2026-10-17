# SPDX-License-Identifier: Apache-2.0
"""仅 CPU 解析/生成静态 CSA；绝不调用 load 或设备执行。"""

import argparse
import json
import re
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from pypto import ir, passes
from pypto.runtime import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--memory-planner", choices=("PYPTO", "DSA_RP", "PTOAS"), default="PYPTO")
    parser.add_argument("--buffer-ir", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"source": str(args.source.resolve()), "device_execution": False,
              "parse_passed": False, "full_codegen_passed": False,
              "runtime_abi_supported": False}
    result["memory_planner"] = args.memory_planner
    result["buffer_ir"] = args.buffer_ir
    try:
        snapshot = args.output / "source.py"
        snapshot.write_text(args.source.read_text())
        result["source_snapshot"] = str(snapshot.resolve())
        program = pl.parse(snapshot.read_text(), filename=str(snapshot.resolve()))
        result.update(parse_passed=True, functions=len(program.functions))
        if not args.parse_only:
            with passes.PassContext([], memory_planner=getattr(passes.MemoryPlanner, args.memory_planner),
                                    enable_buffer_ir=args.buffer_ir):
                compiled = ir.compile(program, **RunConfig(
                    platform="a2a3", save_kernels=True, save_kernels_dir=str(args.output / "kernel"),
                    dump_passes=True).compile_kwargs())
            result.update(full_codegen_passed=True, compiled_directory=str(compiled.output_dir))
            orch_files = list((Path(compiled.output_dir) / "orchestration").glob("*.cpp"))
            code = "\n".join(p.read_text() for p in orch_files)
            result["submission_calls"] = re.findall(r"\brt_submit_\w+\s*\(", code)
            result["physical_kernels"] = [str(p.relative_to(compiled.output_dir))
                                          for p in (Path(compiled.output_dir) / "ptoas").glob("*.cpp")]
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
