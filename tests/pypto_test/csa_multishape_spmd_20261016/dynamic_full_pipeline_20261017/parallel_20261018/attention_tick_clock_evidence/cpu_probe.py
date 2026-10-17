# SPDX-License-Identifier: Apache-2.0
"""包住原完整符号CPU入口，只为这次生成C++添加诊断。"""
import runpy
import sys
from pathlib import Path

MAIN = Path('/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
import dsv4_csa_env
dsv4_csa_env.activate()
from attention_tick_clock import compilation_probe

with compilation_probe(HERE / 'cpu_v1_codegen'):
    runpy.run_path(str(MAIN / 'check_cpu.py'), run_name='__main__')
