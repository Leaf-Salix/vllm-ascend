"""选用冻结算子，复用支持队列分配设备的现有 Sparse 诊断。"""

import importlib.util
import sys
from pathlib import Path

source = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(source / 'tests/pypto_test'))
driver = Path(__file__).resolve().parents[2] / 'dsv4_csa_sparse_diagnostic.py'
spec = importlib.util.spec_from_file_location('sparse_diagnostic', driver)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
