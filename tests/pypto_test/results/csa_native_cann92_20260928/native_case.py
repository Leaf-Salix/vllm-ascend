"""Run the frozen Native-only layer case and record loaded runtime provenance."""

import importlib.metadata
import os
import sys
from pathlib import Path

SOURCE = Path('/data/pyptouser/qinchuanyu/pto-eager/.cache/csa-native-cann92-9d237d33')
sys.path.insert(0, str(SOURCE / 'tests/pypto_test'))
case = importlib.import_module("dsv4_csa_single_layer")

original_write = case.write_json


def write_json(path, data):
    if Path(path).name == 'report.json':
        maps = Path('/proc/self/maps').read_text().splitlines()
        libraries = sorted({line.split()[-1] for line in maps if '.so' in line
                            and ('Ascend/' in line or '/atb/' in line or 'custom_transformer' in line
                                 or 'vllm_ascend' in line or 'torch_npu' in line)})
        data['runtime_provenance'] = {
            'source_commit': '9d237d33',
            'ascend_home': os.environ.get('ASCEND_HOME_PATH'),
            'opp': os.environ.get('ASCEND_OPP_PATH'),
            'custom_opp': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
            'atb_home': os.environ.get('ATB_HOME_PATH'),
            'torch': importlib.metadata.version('torch'),
            'torch_npu': importlib.metadata.version('torch-npu'),
            'libraries': libraries,
            'scope': 'CANN runtime and built-in operators changed; release custom kernels and ATB held fixed',
        }
    original_write(path, data)


case.write_json = write_json
case.main()
