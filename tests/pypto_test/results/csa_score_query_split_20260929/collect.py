"""Reuse the paired full-state and official level-4 checks for query-partitioned AIVs."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    path = ROOT.parent / "csa_score_segment_ub_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("query_split_paired_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.SOURCE_PREFIX = ROOT.parents[4] / ".cache/csa-score-query-split-4ffccb7b"
    module.VARIANT = "pkg:dsv4_csa_score_query_split_4ffccb7b"
    module.TITLE = "长档S6按query分配AIV：真实编译CSA及核内对照"
    module.CHANGE_DESCRIPTION = (
        "长档S6按query分配AIV，每侧三个query且读取两个候选半区；"
        "Cube、排序顺序、两个half根、短档源码路径和任务依赖保持。"
    )
    module.main()
