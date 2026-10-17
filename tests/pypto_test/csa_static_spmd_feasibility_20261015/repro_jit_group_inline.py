# SPDX-License-Identifier: Apache-2.0
"""只在 CPU 验证 JIT inline 的单 Group wrapper 表达，不执行设备或替代 CSA 验证。"""

import argparse
import json
import re
import traceback
from pathlib import Path

import dsv4_csa_env

dsv4_csa_env.activate()

import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit.incore
def cube(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
         c: pl.InOut[pl.Tensor[[384, 16], pl.FP32]]):
    block = pl.tile.get_block_idx()
    left = pl.load(a, [block * 16, 0], [16, 16], target_memory=pl.Mem.Mat)
    right = pl.load(b, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
    value = pl.matmul(left, right)
    c = pl.store(value, [block * 16, 0], c)
    return c


@pl.jit.incore
def vector(x: pl.Tensor[[384, 16], pl.FP32], y: pl.InOut[pl.Tensor[[384, 16], pl.FP32]]):
    pl.func_attr({'split_aiv': True})
    block = pl.tile.get_block_idx()
    lane = pl.tile.get_subblock_idx()
    value = pl.load(x, [block * 16 + lane * 8, 0], [8, 16])
    doubled = pl.add(value, value)
    y = pl.store(doubled, [block * 16 + lane * 8, 0], y)
    return y


@pl.jit.inline
def group_wrapper(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
                  c: pl.InOut[pl.Tensor[[384, 16], pl.FP32]], x: pl.Tensor[[384, 16], pl.FP32],
                  y: pl.InOut[pl.Tensor[[384, 16], pl.FP32]]):
    with pl.cluster(name_hint='csa_group_probe'):
        cube_result = cube(a, b, c)
        vector_result = vector(x, y)
    return c, y


@pl.jit
def entry(a: pl.Tensor[[384, 16], pl.BF16], b: pl.Tensor[[16, 16], pl.BF16],
          c: pl.InOut[pl.Tensor[[384, 16], pl.FP32]], x: pl.Tensor[[384, 16], pl.FP32],
          y: pl.InOut[pl.Tensor[[384, 16], pl.FP32]]):
    with pl.spmd(24, sync_start=True):
        c, y = group_wrapper(a, b, c, x, y)
    return c, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'JIT inline wrapper API 的 CPU 编译探针', 'device_execution': False,
              'full_codegen_passed': False, 'single_mix_submission': False}
    try:
        compiled = entry.compile(config=RunConfig(platform='a2a3', save_kernels=True,
                                                  save_kernels_dir=str(args.output/'kernel'), dump_passes=True))
        result['full_codegen_passed'] = True
        code = '\n'.join(p.read_text() for p in (Path(compiled.output_dir)/'orchestration').glob('*.cpp'))
        calls = re.findall(r'\brt_submit_\w+\s*\(', code)
        result.update(submissions=calls, single_mix_submission=len(calls)==1 and 'MixedKernels' in code,
                      block_num_24='set_block_num(24)' in code, sync_start='set_require_sync_start(true)' in code)
        if not result['single_mix_submission'] or not result['block_num_24'] or not result['sync_start']:
            raise AssertionError('编译完成但没有保留单次 24 组 MIX 提交')
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='traceback'},ensure_ascii=False))


if __name__ == '__main__':
    main()
