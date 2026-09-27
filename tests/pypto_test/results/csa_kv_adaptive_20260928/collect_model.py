"""复用严格入场的两档收集器，保留实际 Native 控制来源。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "csa_source_split_ab_20260927/ordered/collect_model.py"


def main():
    spec = importlib.util.spec_from_file_location("ordered_forward_collector", SOURCE)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.ROOT = ROOT / "model"
    collector.CASES = ((131072, 16), (8192, 40))
    collector.OPERATOR_REVISION = "f1e4cee2 + adaptive fixed-K KV M32/M64; atomic_add=0, original cache, det0"
    collector.main()


if __name__ == "__main__":
    main()
