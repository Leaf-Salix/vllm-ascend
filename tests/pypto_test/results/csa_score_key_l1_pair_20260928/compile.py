"""CPU编译长档双query的Key L1/L0B双缓冲候选，不执行NPU。"""

import os
import sys
from pathlib import Path

SOURCE = Path('/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-score-key-l1-pair-554b3bca')


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
    print('COMPILE_PASS full performance CSA; long pair dedicated Key L1 with L0B prefetch; no device execution')


if __name__ == '__main__':
    main()
