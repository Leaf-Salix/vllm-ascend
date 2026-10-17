# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""逐入口进行真实动态签名CPU后端编译；保留源码与失败，不使用设备。"""

import argparse
import json
import traceback
from pathlib import Path

import dsv4_csa_env

# 先选择既有隔离依赖，再导入编译器；避免测试误用共享安装。
dsv4_csa_env.activate()
import probe
from pypto import passes
from pypto.runtime import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stages", nargs="+", choices=("offsets", "rope", "cache", "count"), default=("offsets", "rope", "cache")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"device_execution": False, "sample_shapes_supplied": False, "stages": {}}
    for name in ("request_geometry.py", "metadata_stages.py", "cache_stages.py"):
        (args.output / name).write_text((probe.SOURCE / name).read_text())
    (args.output / "probe.py").write_text(Path(probe.__file__).read_text())
    try:
        for name in args.stages:
            target = getattr(probe, name + "_entry")
            state = report["stages"][name] = {"parse_passed": False, "full_codegen_passed": False}
            program = target.specialize()
            ir = program.as_python()
            (args.output / (name + "_symbolic.py")).write_text(ir)
            state["parse_passed"] = True
            artifact = target.compile(
                config=RunConfig(
                    platform="a2a3",
                    memory_planner=passes.MemoryPlanner.PYPTO,
                    save_kernels=True,
                    save_kernels_dir=str(args.output / name),
                    dump_passes=True,
                )
            )
            state.update(full_codegen_passed=True, artifact=str(artifact.output_dir))
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
