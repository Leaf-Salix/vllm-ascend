"""使用冻结适配器和私有算子运行独立 Sparse 诊断。"""
import importlib.util
import sys
from pathlib import Path

source = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(source / "tests/pypto_test"))
driver = Path(__file__).with_name("sparse_diagnostic.py")
spec = importlib.util.spec_from_file_location("private_sparse_diagnostic", driver)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
