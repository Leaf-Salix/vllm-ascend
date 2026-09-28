"""复用严格入场的两档收集器，保留实际 Native 控制来源。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "csa_source_split_ab_20260927/ordered/collect_model.py"


def format_json(value, level=0):
    """将位置、样本等标量数组放在同一行，保留全部数据并减少记录体积。"""
    indent = "  " * level
    child_indent = indent + "  "
    if isinstance(value, dict) and value:
        entries = [child_indent + json.dumps(k, ensure_ascii=False) + ": " + format_json(v, level + 1)
                   for k, v in value.items()]
        return "{\n" + ",\n".join(entries) + "\n" + indent + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        entries = [child_indent + format_json(v, level + 1) for v in value]
        return "[\n" + ",\n".join(entries) + "\n" + indent + "]"
    return json.dumps(value, ensure_ascii=False)


def main():
    spec = importlib.util.spec_from_file_location("ordered_forward_collector", SOURCE)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.ROOT = ROOT / "model"
    collector.CASES = ((131072, 16), (8192, 16))
    collector.OPERATOR_REVISION = ("d1f170ff: QLI V2 four-way Top-K + HC input/RMS reuse; prewarmed events; "
                                   "atomic0/mode2/det0")
    collector.main()
    path = collector.ROOT / "forward.json"
    data = json.loads(path.read_text())
    measured = [row for row in data["cases"] if "forward" in row]
    observations = {"cases_with_comparable_timing": len(measured)}
    for label, field, metric in (
        ("lower_forward_mean_cases", "forward", "mean_us"),
        ("lower_forward_p95_cases", "forward", "p95_us"),
        ("lower_forward_max_cases", "forward", "max_us"),
        ("lower_slowest_rank_mean_cases", "slowest_rank_forward", "mean_us"),
    ):
        observations[label] = sum(row[field]["pto"][metric] < row[field]["native"][metric] for row in measured)
    observations["scope"] = "记录性能观测数量，与token/DSpark通过分开；不能用功能PASS代替性能收益。"
    data["performance_observations"] = observations
    path.write_text(format_json(data) + "\n")
    print(observations)


if __name__ == "__main__":
    main()
