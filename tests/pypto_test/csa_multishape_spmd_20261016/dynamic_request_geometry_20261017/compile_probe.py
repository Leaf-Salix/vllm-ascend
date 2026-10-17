# SPDX-License-Identifier: Apache-2.0
"""隔离工具链内解析并完整 CPU 编译动态探针，不创建 NPU runtime。"""

import argparse
import json
import traceback
from pathlib import Path

import dsv4_csa_env


def main():
    # 工具链激活后才载入 PyPTO 和同源探针，避免导入共享安装。
    dsv4_csa_env.activate()
    import probe
    from pypto import passes
    from pypto.runtime import CompileOptions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    result = {"source": str(probe.SOURCE), "parse_passed": False, "full_cpu_lowering_passed": False,
              "device_execution": False, "signature_mode": True, "sample_shapes_supplied": False}
    (args.output / "request_geometry.py").write_text(probe.SOURCE.read_text())
    try:
        parsed = probe.entry.specialize()
        ir_source = parsed.as_python()
        (args.output / "parsed.py").write_text(ir_source)
        result["parse_passed"] = True
        result["dynamic_symbols_present"] = {
            name: name in ir_source for name in (
                "CSA_GEOMETRY_QUERY_BOUNDS", "CSA_GEOMETRY_REQUESTS", "CSA_GEOMETRY_PROBE_TOKENS"
            )
        }
        if not all(result["dynamic_symbols_present"].values()):
            raise AssertionError("解析后的 program 丢失动态轴")
        artifact = probe.entry._resolve_kernel_artifact(
            (), {"config": CompileOptions(platform="a2a3", output_dir=str(args.output / "kernel"),
                                           dump_passes=True, memory_planner=passes.MemoryPlanner.PYPTO)},
            allow_signature_mode=True,
        )
        result.update(full_cpu_lowering_passed=True, output_dir=str(artifact.output_dir))
    except Exception as error:
        result.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "traceback"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
