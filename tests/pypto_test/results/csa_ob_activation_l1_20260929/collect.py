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
    module.TITLE = "NZ O-B小中档完整激活L1复用，权重保留分段流水"
    module.TARGETS = {"aic": "proj_b_mm"}
    module.DESCRIPTION = (
        "长B16使用ROW96的新A复用路径；短B24为ROW128不变路径控制。"
        "量化及整数累加、任务/worker数、所有依赖和调度标志不变。"
        "不把短档控制的波动当作新核内优化收益。"
    )
    module.main()
