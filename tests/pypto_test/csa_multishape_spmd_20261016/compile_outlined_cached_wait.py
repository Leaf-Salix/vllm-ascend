# SPDX-License-Identifier: Apache-2.0
"""CPU诊断：全阶段独立代码生成与现选100tick等待能否安全组合。

仅复用现有编译扩展，不修改依赖或运行中的driver。全部阶段独立时才允许
清理外层原数学事件；阶段自身事件与业务数据维护保留。未测设备性能。
"""

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

from pypto import passes
from pypto.runtime import CompileOptions

from signal_polling import audit_wait_artifact, bypass_signal_wait
from stage_codegen import independent_stage_codegen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"case": str(args.case.resolve()), "full_codegen_passed": False,
              "device_executed": False, "eligible_for_700us_target": False}
    try:
        snapshot = args.output / "source.py"
        snapshot.write_text((args.case / "entry_jit.py").read_text())
        layout = json.loads((args.case / "layout.json").read_text())
        spec = importlib.util.spec_from_file_location("csa_outlined_cached_cpu", snapshot)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with independent_stage_codegen(layout, args.output / "stages", selected=("*",),
                                       compact_params=True, noinline=True):
            with bypass_signal_wait(args.output / "signal_polling", 100, cached=True):
                artifact = module.entry._resolve_kernel_artifact((), {"config": CompileOptions(
                    platform="a2a3", output_dir=str(args.output / "kernel"), dump_passes=True,
                    memory_planner=passes.MemoryPlanner.PYPTO)}, allow_signature_mode=True)
        result["full_codegen_passed"] = True
        result["wait_audit"] = audit_wait_artifact(artifact.output_dir,
                                                 args.output / "signal_polling", 100, cached=True)
        changes = json.loads((args.output / "stages/audit.json").read_text())
        boundaries = [row for row in changes if row.get("outer_math_fully_outlined")]
        if {row["role"] for row in boundaries} != {"aic", "aiv"}:
            raise ValueError("两种核的外层数学事件没有完整清理")
        result["outer_event_audit"] = boundaries
        result["outlined_stages"] = sum("stage" in row for row in changes)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
