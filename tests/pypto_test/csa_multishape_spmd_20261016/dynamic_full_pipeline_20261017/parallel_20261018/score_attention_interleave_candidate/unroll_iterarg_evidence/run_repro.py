# SPDX-License-Identifier: Apache-2.0
"""两份最小动态IR只运行Inline/Unroll/CtrlFlow/SSA，不lower整CSA或生成设备二进制。"""

import argparse
import json
import re
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    overlay = Path("/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-ultra-indexer-20261018/tests/pypto_test/jit_tile_parameter_overlay_20261018/pypto_overlay")
    for finder in list(sys.meta_path):
        if "pypto.pypto_core" in getattr(finder, "known_wheel_files", {}):
            sys.meta_path.remove(finder)
    sys.path.insert(0, str(overlay / "python"))
    import pypto
    from pypto import passes
    from pypto.ir.printer import python_print
    from scope_kernels import alias_entry, direct_entry

    assert overlay in Path(pypto.__file__).resolve().parents
    report = {
        "cpu_only": True, "whole_csa_compiled": False, "npu_execution": False,
        "sample_shapes_supplied": False, "source": pypto.__file__,
        "binary": str(Path(pypto.pypto_core.__file__).resolve()), "cases": {},
    }
    for name, entry in (("direct", direct_entry), ("alias", alias_entry)):
        out = args.output / name
        out.mkdir()
        case = {"ssa_passed": False, "stage": "specialize", "passes": []}
        report["cases"][name] = case
        try:
            program = entry.specialize()
            steps = (
                ("00_frontend", None),
                ("01_inline", passes.inline_functions()),
                ("02_unroll", passes.unroll_loops()),
                ("03_control_flow", passes.ctrl_flow_transform()),
                ("04_ssa", passes.convert_to_ssa()),
            )
            for stage, transform in steps:
                case["stage"] = stage
                if transform is not None:
                    program = transform(program)
                text = python_print(program)
                path = out / (stage + ".py")
                path.write_text(text)
                case["passes"].append({
                    "stage": stage, "path": str(path), "bytes": path.stat().st_size,
                    "free_var_count": text.count("__FREE_VAR"),
                    "free_var_names": sorted(set(re.findall(r"\b\w+__FREE_VAR\b", text))),
                })
            case["stage"] = "verify_ssa"
            properties = passes.IRPropertySet()
            properties.insert(passes.IRProperty.SSAForm)
            passes.run_verifier(properties)(program)
            case["ssa_passed"] = True
        except Exception as error:
            detail = out / "error_full.txt"
            message = str(error)
            detail.write_text(message + "\n")
            (out / "traceback_stack.txt").write_text("".join(traceback.format_tb(error.__traceback__)))
            case.update(error_type=type(error).__name__, error_preview=message[:2048],
                        error_path=str(detail), error_bytes=detail.stat().st_size)
        (out / "result.json").write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
    report["hypothesis_confirmed"] = (
        not report["cases"]["direct"]["ssa_passed"]
        and report["cases"]["direct"]["stage"] == "verify_ssa"
        and report["cases"]["alias"]["ssa_passed"]
        and report["cases"]["direct"]["passes"][1]["free_var_count"] == 0
        and report["cases"]["direct"]["passes"][2]["free_var_count"] > 0
        and all(p["free_var_count"] == 0 for p in report["cases"]["alias"]["passes"])
    )
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["hypothesis_confirmed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
