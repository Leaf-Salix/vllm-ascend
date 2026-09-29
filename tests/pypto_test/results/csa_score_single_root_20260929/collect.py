"""完整核对单根候选的状态、正式CSA及Score/merge四窗核时。"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / "csa_sparse_first_pv_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("single_root_collection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = "长档按query分工、UB连续排序及单根发布"
    module.TARGETS = {"aic": "indexer_score_topk_native_pair_aic",
                      "aiv": "indexer_score_topk_native_pair_aiv", "merge": "indexer_topk_query_merge"}
    module.DESCRIPTION = (
        "只改变长S6的候选组织与Top-K结构；FP16/Cube缩放和量化不变。"
        "同分顺序可能不同，仍先用八类完整状态零容差揭示差异，不自动放宽验收。"
    )
    module.main()


if __name__ == "__main__":
    main()
