# SPDX-License-Identifier: Apache-2.0
"""仅CPU编译实验pair计划的编排与设备C++，不启动Worker或NPU。"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_source = args.output / "source.py"
    frozen_source.write_text(args.source.read_text())
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output.resolve() / "build")
    os.environ["PYPTO_CACHE_DIR"] = str(args.output.resolve() / "cache")
    from pypto import passes
    from pypto.runtime import CompileOptions
    from pypto.runtime.kernel_compiler import KernelCompiler
    from pypto.runtime.pto_isa import ensure_pto_isa_root

    result = {"device_execution": False, "source": str(args.source.resolve()), "passed": False}
    try:
        spec = importlib.util.spec_from_file_location("hca_pair_plan_candidate", frozen_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.pair_plan_test.compile(config=CompileOptions(platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO))
        compiler = KernelCompiler(platform="a2a3")
        orchestration = sorted((args.output / "build").glob("*/orchestration/*.cpp"))
        kernels = sorted((args.output / "build").glob("*/kernels/*/*.cpp"))
        if not orchestration or not kernels:
            raise RuntimeError("没有生成所需C++，不能视为完整编译通过")
        result["binaries"] = []
        for source in orchestration:
            binary = compiler.compile_orchestration("tensormap_and_ringbuffer", str(source))
            source.with_suffix(".so").write_bytes(binary)
            result["binaries"].append({"source": str(source), "bytes": len(binary)})
        for source in kernels:
            binary = compiler.compile_incore(str(source), core_type=source.parent.name,
                                              pto_isa_root=str(ensure_pto_isa_root()),
                                              runtime_name="tensormap_and_ringbuffer")
            source.with_suffix(".o").write_bytes(binary)
            result["binaries"].append({"source": str(source), "bytes": len(binary)})
        result["passed"] = True
    except Exception:
        result["traceback"] = traceback.format_exc()
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
