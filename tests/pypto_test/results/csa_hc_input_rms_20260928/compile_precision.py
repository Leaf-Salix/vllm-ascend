"""仅编译共享HC抽取后的精度版完整入口，不执行设备代码。"""
import os
import sys
from pathlib import Path

source = Path('/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-hc-input-rms-9edb8dfe')
sys.path.insert(0, str(source / 'tests/pypto_test'))
from dsv4_csa_env import activate

os.environ['VLLM_ASCEND_ENABLE_NZ'] = '2'
activate()
from pypto.runtime import RunConfig
from vllm_ascend.ops.pypto.deepseek_v4_flash_dspark.decode_csa import decode_csa_tp1_layer_test

folder = Path(__file__).resolve().parent / 'compiled_precision'
compiled = decode_csa_tp1_layer_test.compile(config=RunConfig(
    platform='a2a3', save_kernels=True, save_kernels_dir=str(folder)))
compiled.load()
print('COMPILE_PASS precision full CSA; no device execution')
