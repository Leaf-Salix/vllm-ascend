# SPDX-License-Identifier: Apache-2.0
"""Compile an isolated candidate using saved descriptors and CPU meta tensors."""

import argparse
import importlib
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    report = {"passed": False, "device_execution": False, "source": str(args.source)}
    try:
        import torch
        from pypto import passes
        from pypto.ir.printer import python_print
        from pypto.runtime import CompileOptions
        from source_loader import load_package

        package = load_package(args.source, "capacity_compile")
        entry = importlib.import_module(package.name + ".entry").decode_csa_spmd
        description = json.loads(args.descriptors.read_text())
        tensors = {
            name: torch.empty(row["shape"], dtype=getattr(torch, row["dtype"].removeprefix("torch.")), device="meta")
            for name, row in description["buffers"].items()
        }
        arguments = dict(tensors, **description["policy"])
        program = entry.specialize(**arguments)
        (args.output / "specialized.py").write_text(python_print(program))
        report["parse_passed"] = True
        if not args.parse_only:
            arguments["config"] = CompileOptions(
                platform="a2a3",
                output_dir=str(args.output / "kernel"),
                dump_passes=True,
                memory_planner=passes.MemoryPlanner.PYPTO,
            )
            artifact = entry._resolve_kernel_artifact((), arguments)
            report["artifact"] = str(artifact.output_dir)
            report["full_codegen_passed"] = True
        report["passed"] = True
    except BaseException as exc:
        report.update(
            error=str(exc),
            error_span=getattr(exc, "span", None),
            traceback=traceback.format_exc(),
        )
    (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "traceback"}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
