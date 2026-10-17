# SPDX-License-Identifier: Apache-2.0
"""只做动态发布helper的CPU full backend，保存每次源快照和失败，不运行NPU。"""

import argparse
import json
import re
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stages", nargs="+", default=["pair_one_balanced", "pair_two_balanced", "pair_two_sequential", "single_leaf"]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    source = here.parents[3] / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
    for name in (
        "indexer_publish_stages.py",
        "indexer_group_stages.py",
        "indexer_topk_stages.py",
        "request_geometry.py",
    ):
        (args.output / name).write_text((source / name).read_text())
    for name in ("cpu_probes.py", "check_cpu.py"):
        (args.output / name).write_text((here / name).read_text())
    report = {"device_execution": False, "sample_shapes_supplied": False, "performance_measured": False, "stages": {}}
    try:
        import dsv4_csa_env

        dsv4_csa_env.activate()
        import cpu_probes
        from pypto import passes
        from pypto.runtime import RunConfig

        for name in args.stages:
            record = report["stages"][name] = {"passed": False}
            try:
                entry = getattr(cpu_probes, name)
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
                directory = Path(artifact.output_dir)
                metadata = json.loads((directory / "compiled_meta.json").read_text())
                dimensions = {}
                for path in (directory / "kernels").rglob("*.cpp"):
                    dimensions[str(path.relative_to(directory))] = sorted(
                        set(re.findall(r"int64_t (CSA_\w+) = .*?->shapes\[\d+\]\);", path.read_text()))
                    )
                flattened = {dim for dims in dimensions.values() for dim in dims}
                assert {"CSA_INDEX_GROUP_T", "CSA_GEOMETRY_QUERY_BOUNDS", "CSA_GEOMETRY_REQUESTS"} <= flattened
                assert any(-1 in p["shape"] for p in metadata["params"])
                record.update(
                    passed=True,
                    full_codegen_passed=True,
                    artifact=str(directory),
                    dynamic_dimensions=dimensions,
                    parameter_shapes=[p["shape"] for p in metadata["params"]],
                )
            except BaseException as error:
                record.update(error=str(error), traceback=traceback.format_exc())
            (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    report["passed"] = bool(report["stages"]) and all(s["passed"] for s in report["stages"].values())
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
