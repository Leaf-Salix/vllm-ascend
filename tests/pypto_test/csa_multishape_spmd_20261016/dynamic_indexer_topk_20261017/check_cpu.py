# SPDX-License-Identifier: Apache-2.0
"""仅按动态签名完整CPU编译四段纯排序/归并，保留实际失败；不执行NPU或读取权重。"""

import argparse
import json
import re
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["merge2", "roots", "half_leaf", "segment"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import dsv4_csa_env

    dsv4_csa_env.activate()
    import cpu_probes
    from pypto import passes
    from pypto.ir.printer import python_print
    from pypto.runtime import CompileOptions

    here = Path(__file__).resolve().parent
    source = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/indexer_topk_stages.py"
    (args.output / "indexer_topk_stages.py").write_text(source.read_text())
    (args.output / "cpu_probes.py").write_text((here / "cpu_probes.py").read_text())
    result = {
        "scope": "四个独立源级排序/归并阶段完整CPU编译，尚非阶段融合",
        "device_execution": False,
        "weight_values_loaded": False,
        "stages": [],
    }
    failed = False
    for stage in args.stages:
        target = args.output / stage
        target.mkdir()
        row = {"stage": stage, "status": "FAIL", "full_codegen_passed": False}
        result["stages"].append(row)
        try:
            kernel = getattr(cpu_probes, stage + "_probe")
            (target / "symbolic.py").write_text(python_print(kernel.specialize()))
            artifact = kernel._resolve_kernel_artifact(
                (),
                {
                    "config": CompileOptions(
                        platform="a2a3",
                        output_dir=str(target / "kernel"),
                        dump_passes=True,
                        memory_planner=passes.MemoryPlanner.PYPTO,
                    )
                },
                allow_signature_mode=True,
            )
            output = Path(artifact.output_dir)
            kernels = list((output / "kernels").rglob("*.cpp"))
            dynamic = {}
            for path in kernels:
                text = path.read_text()
                dynamic[str(path.relative_to(output))] = re.findall(
                    r"int64_t (CSA_INDEX_\w+) = .*?->shapes\[\d+\]\);", text
                )
            assert kernels and any(dynamic.values()), "最终CPP必须保留运行时动态Tensor维度"
            row.update(status="PASS", full_codegen_passed=True, artifact=str(output), dynamic_cpp_dimensions=dynamic)
        except Exception as exc:
            failed = True
            row.update(error=str(exc), traceback=traceback.format_exc())
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in row.items() if key != "traceback"}, ensure_ascii=False), flush=True
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
