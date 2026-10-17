# SPDX-License-Identifier: Apache-2.0
"""两段动态 Indexer cache 的 CPU 后端编译与动态 ABI 检查；不运行设备。"""

import argparse
import json
import re
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stages", nargs="+", default=["indexer_compact_quant_cache", "indexer_compact_scale_commit"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import dsv4_csa_env

    dsv4_csa_env.activate()
    import cpu_probes
    from pypto import passes
    from pypto.runtime import RunConfig

    here = Path(__file__).resolve().parent
    source = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    for name in ("request_geometry.py", "indexer_cache_stages.py"):
        (args.output / name).write_text((source / name).read_text())
    for name in ("cpu_probes.py", "check_cpu.py"):
        (args.output / name).write_text((here / name).read_text())
    report = {"device_execution": False, "performance_measured": False, "sample_shapes_supplied": False, "stages": {}}
    for name in args.stages:
        entry = getattr(cpu_probes, name + "_probe")
        record = report["stages"][name] = {"passed": False}
        try:
            (args.output / (name + "_symbolic.py")).write_text(entry.specialize().as_python())
            artifact = entry.compile(
                config=RunConfig(
                    platform="a2a3",
                    memory_planner=passes.MemoryPlanner.PYPTO,
                    save_kernels=True,
                    save_kernels_dir=str(args.output / name),
                    dump_passes=True,
                )
            )
            artifact_path = Path(artifact.output_dir)
            metadata = json.loads((artifact_path / "compiled_meta.json").read_text())
            assert any(-1 in p["shape"] for p in metadata["params"])
            dimensions = {}
            for path in (artifact_path / "kernels").rglob("*.cpp"):
                dimensions[str(path.relative_to(artifact_path))] = re.findall(
                    r"int64_t (CSA_(?:INDEX_CACHE|GEOMETRY)_\w+) = .*?->shapes\[\d+\]\);", path.read_text()
                )
            assert any(dimensions.values()), "实际核源码必须保留动态descriptor维度"
            record.update(
                passed=True, full_codegen_passed=True, artifact=str(artifact_path), dynamic_cpp_dimensions=dimensions
            )
        except Exception as exc:
            record.update(error=str(exc), traceback=traceback.format_exc())
        report["passed"] = all(row["passed"] for row in report["stages"].values())
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
