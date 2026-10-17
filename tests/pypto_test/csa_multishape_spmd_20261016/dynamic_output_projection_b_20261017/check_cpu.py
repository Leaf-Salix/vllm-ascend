# SPDX-License-Identifier: Apache-2.0
"""只进行动态ABI的CPU full lowering；逐次保留源码、失败和产物，不执行NPU。"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path


def audit_artifact(directory, name):
    """兼容两种真实动态ABI：Tensor descriptor读取，或同轮dim作为Scalar传入核。"""
    metadata = json.loads((directory / "compiled_meta.json").read_text())
    assert any(-1 in parameter["shape"] for parameter in metadata["params"])
    dimensions, scalar_inputs = {}, {}
    for path in (directory / "kernels").rglob("*.cpp"):
        source = path.read_text()
        key = str(path.relative_to(directory))
        dimensions[key] = sorted(set(re.findall(
            r"int64_t (CSA_OUTPUT_B_\w+) = .*?->shapes\[\d+\]\);", source)))
        scalar_inputs[key] = sorted(set(re.findall(r"// Unpack scalar: (CSA_OUTPUT_B_\w+)", source)))
    if name == "dequant":
        assert any(set(("CSA_OUTPUT_B_T", "CSA_OUTPUT_B_MM_ROWS", "CSA_OUTPUT_B_SCALE_ROWS")) <= set(dims)
                   for dims in dimensions.values())
        actual_t_binding = "核内直接读取out descriptor的T；MM/SCALE各自独立读取"
    else:
        orchestration = "\n".join(path.read_text() for path in (directory / "orchestration").glob("*.cpp"))
        assert re.search(r"CSA_OUTPUT_B_T = .*?shapes\[0\]", orchestration)
        assert ".add_scalar(CSA_OUTPUT_B_T)" in orchestration
        assert any("CSA_OUTPUT_B_T" in values for values in scalar_inputs.values())
        assert any("CSA_OUTPUT_B_MM_ROWS" in dims for dims in dimensions.values())
        actual_t_binding = "外层从本轮residual descriptor读取T后动态Scalar传入AIC，未用sample shape"
        expected_layout = "ND" if os.environ.get("VLLM_ASCEND_ENABLE_NZ") == "0" else "NZ"
        assert next(p for p in metadata["params"] if p["name"] == "weight")["layout"] == expected_layout
    return dict(dynamic_cpp_dimensions=dimensions, dynamic_scalar_inputs=scalar_inputs,
                actual_t_binding=actual_t_binding, parameter_shapes=[p["shape"] for p in metadata["params"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["ob32", "ob96", "ob128", "dequant"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    source = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/output_projection_b_stages.py"
    for path in (source, here / "check_cpu.py", here / "cpu_probes.py"):
        (args.output / path.name).write_text(path.read_text())
    report = {"device_execution": False, "sample_shapes_supplied": False, "performance_measured": False,
              "weight_nz_mode": os.environ.get("VLLM_ASCEND_ENABLE_NZ"), "stages": {}}
    try:
        import dsv4_csa_env
        dsv4_csa_env.activate()
        import cpu_probes
        from pypto import passes
        from pypto.runtime import RunConfig
        assert "vllm_ascend.ops.pypto.deepseek_v4_flash_csa.nz_mode" not in sys.modules
        for name in args.stages:
            record = report["stages"][name] = {"passed": False}
            try:
                entry = getattr(cpu_probes, name)
                (args.output / (name + "_symbolic.py")).write_text(entry.specialize().as_python())
                artifact = entry.compile(config=RunConfig(
                    platform="a2a3", memory_planner=passes.MemoryPlanner.PYPTO,
                    save_kernels=True, save_kernels_dir=str(args.output / name), dump_passes=True,
                ))
                directory = Path(artifact.output_dir)
                record.update(passed=True, full_codegen_passed=True, artifact=str(directory),
                              **audit_artifact(directory, name))
            except BaseException as error:
                record.update(error=str(error), traceback=traceback.format_exc())
            (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({k:v for k,v in record.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    report["passed"] = bool(report["stages"]) and all(s["passed"] for s in report["stages"].values())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
