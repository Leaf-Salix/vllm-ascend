# SPDX-License-Identifier: Apache-2.0
"""仅编译动态签名后处理，保存源码与失败；不执行NPU或读取模型权重。"""

import argparse
import json
import re
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import dsv4_csa_env

    dsv4_csa_env.activate()
    from cpu_probe import compressor_output_probe
    from pypto import passes
    from pypto.ir.printer import python_print
    from pypto.runtime import CompileOptions

    here = Path(__file__).resolve().parent
    package = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    for name in ("compressor_output_stages", "metadata_stages", "request_geometry"):
        (args.output / f"{name}.py").write_text((package / f"{name}.py").read_text())
    (args.output / "cpu_probe.py").write_text((here / "cpu_probe.py").read_text())
    result = {
        "scope": "主compressor请求驱动RMS/RoPE/cache后处理的独立CPU后端",
        "device_execution": False,
        "weight_values_loaded": False,
        "full_codegen_passed": False,
    }
    try:
        (args.output / "symbolic.py").write_text(python_print(compressor_output_probe.specialize()))
        artifact = compressor_output_probe._resolve_kernel_artifact(
            (),
            {
                "config": CompileOptions(
                    platform="a2a3",
                    output_dir=str(args.output / "kernel"),
                    dump_passes=True,
                    memory_planner=passes.MemoryPlanner.PYPTO,
                )
            },
            allow_signature_mode=True,
        )
        output = Path(artifact.output_dir)
        kernels = list((output / "kernels").rglob("*.cpp"))
        dynamic = {
            str(path.relative_to(output)): re.findall(r"// Extract dynamic dim: (\w+)", path.read_text())
            for path in kernels
        }
        assert kernels and any(dynamic.values()), "CPP必须保留动态维度"
        result.update(full_codegen_passed=True, artifact=str(output), dynamic_cpp_dimensions=dynamic)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    if not result["full_codegen_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
