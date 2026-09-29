"""沿完整状态与官方四窗收集流程，单列O-B核时。"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    path = ROOT.parent / "csa_sparse_first_pv_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("ob_activation_collection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = "NZ O-B激活复用并保持原K128双缓冲"
    module.TARGETS = {"aic": "proj_b_mm"}
    module.DESCRIPTION = (
        "长短B16均覆盖ROW96的A复用，原L0 K128双缓冲保持。"
        "量化及整数累加、任务/worker数、所有依赖和调度标志不变。"
        "量化与整数归约保持，核内与CSA分别按8:2评估。"
    )
    module.main()
