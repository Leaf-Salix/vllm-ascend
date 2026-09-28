"""CPU编译已经分别通过单卡的B4/B8预取组合，不执行NPU。"""

import os
import sys
from pathlib import Path

SOURCE = Path('/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-key-prefetch-combined-d9c7a147')


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
        save_kernels_dir=str(Path(__file__).resolve().parent / 'combined_compiled')))
    compiled.load()
    print('COMPILE_PASS combined B4 Key L1/L0B and B8 Key L1-only; no device execution')


if __name__ == '__main__':
    main()
