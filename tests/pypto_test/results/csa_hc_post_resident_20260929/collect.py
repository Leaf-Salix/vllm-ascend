"""复用状态和官方DFX收集流程，仅把核内目标设为HC_post。"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    path = ROOT.parent / "csa_sparse_first_pv_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("hc_resident_collection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = "HC_post残差行常驻UB，避免重复读取与BF16转换"
    module.TARGETS = {"aiv": "hc_post"}
    module.DESCRIPTION = "只复用每token四行FP32残差；post*x及0/1/2/3乘加顺序、BF16 RINT保持。"
    module.main()
