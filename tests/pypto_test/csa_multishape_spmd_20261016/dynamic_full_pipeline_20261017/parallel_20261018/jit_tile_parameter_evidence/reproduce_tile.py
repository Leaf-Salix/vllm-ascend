# SPDX-License-Identifier: Apache-2.0
"""最小 JIT Tile 入参/返回复现：仅 CPU 特化，可选生成物编译。"""

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--case', choices=('tile', 'state', 'state_full', 'inout', 'constexpr'), default='tile')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    overlay = here / ('pypto_reference' if args.reference else 'pypto_overlay')
    args.output.mkdir(parents=True, exist_ok=False)
    for finder in list(sys.meta_path):
        if 'pypto.pypto_core' in getattr(finder, 'known_wheel_files', {}):
            sys.meta_path.remove(finder)
    sys.path.insert(0, str(overlay / 'python'))

    import pypto
    from pypto.ir.printer import python_print
    from pypto.jit.specializer import Specializer
    tile_entry = importlib.import_module(f'{args.case}_kernels').tile_entry

    assert overlay in Path(pypto.__file__).resolve().parents
    original_specialize = Specializer.specialize

    def capture_specialize(self):
        source = original_specialize(self)
        (args.output / 'generated_preparse.py').write_text(source)
        return source

    Specializer.specialize = capture_specialize
    report = {
        'passed': False, 'stage': 'specialize',
        'case': args.case,
        'pypto_source': pypto.__file__, 'binary': pypto.pypto_core.__file__,
        'dynamic_tensor_rows': True, 'sample_tensors_supplied': False,
        'npu_execution': False, 'compiled': False,
    }
    try:
        program = tile_entry.specialize()
        (args.output / 'pre_pass.py').write_text(python_print(program))
        report['specialized'] = True
        if args.compile:
            report['stage'] = 'compile'
            from pypto import ir

            compiled = ir.compile(
                program, output_dir=str(args.output / 'kernel'),
                platform='a2a3', strategy=ir.OptimizationStrategy.Default,
            )
            report['compiled'] = True
            report['artifact_type'] = type(compiled).__name__
        report['passed'] = True
    except Exception as error:
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        (args.output / 'traceback.txt').write_text(traceback.format_exc())
    finally:
        Specializer.specialize = original_specialize
        (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
