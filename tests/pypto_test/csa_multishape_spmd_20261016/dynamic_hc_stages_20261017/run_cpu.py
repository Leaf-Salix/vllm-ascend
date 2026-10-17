# SPDX-License-Identifier: Apache-2.0
"""只做动态数学 helper 的 CPU lowering/编译，不初始化设备。"""

import argparse
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["widen", "project", "reduce", "gates", "comb", "mix_norm"])
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import dsv4_csa_env

    dsv4_csa_env.activate()
    import cpu_probes
    from pypto import passes
    from pypto.runtime import CompileOptions, RunConfig

    result = {"scope": "动态源码数学阶段CPU编译，无设备执行", "device_execution": False, "stages": []}
    failed = False
    for stage in args.stages:
        row = {"stage": stage, "dynamic_signature": True, "full_codegen_requested": args.full}
        result["stages"].append(row)
        output = args.output / stage
        output.mkdir()
        try:
            kernel = getattr(cpu_probes, stage + "_probe")
            if args.full:
                artifact = kernel._resolve_kernel_artifact(
                    (),
                    {
                        "config": CompileOptions(
                            platform="a2a3",
                            output_dir=str(output / "kernel"),
                            dump_passes=True,
                            memory_planner=passes.MemoryPlanner.PYPTO,
                        )
                    },
                    allow_signature_mode=True,
                )
                row.update(status="PASS", full_codegen_passed=True, output=str(artifact.output_dir))
            else:
                program = kernel.lower(config=RunConfig(platform="a2a3"))
                (output / "lowered.py").write_text(str(program))
                row.update(status="PASS", frontend_lowering_passed=True, full_codegen_passed=False)
        except Exception as exc:
            row.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
            failed = True
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "traceback"}, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
