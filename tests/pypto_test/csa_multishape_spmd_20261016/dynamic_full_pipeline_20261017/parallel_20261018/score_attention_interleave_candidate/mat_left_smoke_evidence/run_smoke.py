# SPDX-License-Identifier: Apache-2.0
"""只运行一个Mat/Left最小CPU lowering，不提供样例Tensor或调用NPU。"""

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", choices=("mat_left_kernels", "mat_left_kernels_v2"), default="mat_left_kernels")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    overlay = Path(__file__).resolve().parent.parent / "jit_tile_parameter_overlay_20261018/pypto_overlay"
    for finder in list(sys.meta_path):
        if "pypto.pypto_core" in getattr(finder, "known_wheel_files", {}):
            sys.meta_path.remove(finder)
    sys.path.insert(0, str(overlay / "python"))
    import pypto
    from pypto import ir
    from pypto.ir.printer import python_print
    from pypto.jit.specializer import Specializer
    tile_entry = importlib.import_module(args.module).tile_entry

    assert overlay in Path(pypto.__file__).resolve().parents
    original = Specializer.specialize

    def capture(self):
        source = original(self)
        (args.output / "generated_preparse.py").write_text(source)
        return source

    Specializer.specialize = capture
    report = dict(passed=False, stage="specialize", fixture=args.module, device_execution=False, sample_shapes_supplied=False,
                  pypto_source=pypto.__file__, binary=str(Path(pypto.pypto_core.__file__).resolve()))
    try:
        program = tile_entry.specialize()
        (args.output / "pre_pass.py").write_text(python_print(program))
        report.update(specialized=True, stage="compile")
        result = ir.compile(program, output_dir=str(args.output / "kernel"),
                            platform="a2a3", strategy=ir.OptimizationStrategy.Default)
        report.update(passed=True, artifact_type=type(result).__name__)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        (args.output / "traceback.txt").write_text(traceback.format_exc())
    finally:
        Specializer.specialize = original
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
