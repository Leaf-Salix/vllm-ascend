"""CPU 编译已保留的 Top-K UB 与 QR 尾修正版组合，不执行设备代码。"""

import os
import sys
from pathlib import Path

SOURCE = Path('/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-ub-combined-2d2f9ca0')


def main():
    sys.path.insert(0, str(SOURCE / 'tests/pypto_test'))
    from dsv4_csa_env import activate

    os.environ['VLLM_ASCEND_ENABLE_NZ'] = '2'
    os.environ['PTO_CSA_VARIANT'] = 'performance'
    os.environ['VLLM_ASCEND_PTO_CSA_ATOMIC_ADD'] = '0'
    activate()
    from pypto.runtime import RunConfig

    from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark_perf.decode_csa import decode_csa_tp1_layer_test

    compiled = decode_csa_tp1_layer_test.compile(config=RunConfig(
        platform='a2a3', save_kernels=True,
        save_kernels_dir=str(Path(__file__).resolve().parent / 'compiled')))
    compiled.load()
    print('COMPILE_PASS full performance CSA 2d2f9ca0; Top-K UB and fixed QR UB; no device execution')


if __name__ == '__main__':
    main()
