# SPDX-License-Identifier: Apache-2.0
"""固定已有 Tile IR 后只运行 DSA_RP 分配；CPU 诊断，不是生产编译路径。"""

import argparse
import json
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from pypto import backend, ir, passes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"source": str(args.source.resolve()), "device_execution": False,
              "full_codegen_passed": False, "allocation_passed": False,
              "scope": "固定 MaterializeSemanticAliases 产物，仅 DSA_RP 地址分配"}
    try:
        source = args.source.read_text()
        (args.output / "source.py").write_text(source)
        program = pl.parse(source, filename=str(args.source.resolve()))
        backend.set_backend_type(backend.BackendType.Ascend910B)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP):
            allocated = passes.allocate_memory_addr()(program)
        (args.output / "allocated.py").write_text(ir.python_print(allocated))
        result["allocation_passed"] = True
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
