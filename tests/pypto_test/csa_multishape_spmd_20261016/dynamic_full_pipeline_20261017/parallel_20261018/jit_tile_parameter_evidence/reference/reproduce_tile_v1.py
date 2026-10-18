# SPDX-License-Identifier: Apache-2.0
"""最小 JIT Tile 入参/返回复现：仅 CPU 特化，可选生成物编译。"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    overlay = here / 'pypto_overlay'
    args.output.mkdir(parents=True, exist_ok=False)
    for finder in list(sys.meta_path):
        if 'pypto.pypto_core' in getattr(finder, 'known_wheel_files', {}):
            sys.meta_path.remove(finder)
    sys.path.insert(0, str(overlay / 'python'))

    import pypto
    import pypto.language as pl
    from pypto.ir.printer import python_print
    from pypto.jit.specializer import Specializer

    assert overlay in Path(pypto.__file__).resolve().parents
    dynamic_rows = pl.dynamic('TILE_PARAM_DYNAMIC_ROWS')

    @pl.jit.inline(auto_scope=False)
    def keep_tile(tile: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec]) -> pl.Tile:
        return tile

    @pl.jit.incore
    def tile_kernel(
        source: pl.Tensor[[dynamic_rows, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[dynamic_rows, 64], pl.FP32]],
    ):
        rows = pl.min(pl.tensor.dim(source, 0), 8)
        tile = pl.load(source, [0, 0], [8, 64], valid_shape=[rows, 64])
        same_tile = keep_tile(tile)
        pl.store(same_tile, [0, 0], output)
        return output

    @pl.jit
    def tile_entry(
        source: pl.Tensor[[dynamic_rows, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[dynamic_rows, 64], pl.FP32]],
    ):
        return tile_kernel(source, output)

    original_specialize = Specializer.specialize

    def capture_specialize(self):
        source = original_specialize(self)
        (args.output / 'generated_preparse.py').write_text(source)
        return source

    Specializer.specialize = capture_specialize
    report = {
        'passed': False, 'stage': 'specialize',
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
