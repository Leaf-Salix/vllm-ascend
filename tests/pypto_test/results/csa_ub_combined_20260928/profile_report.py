"""收集当前组合的相邻CSA、模型区间及同轮Native七三加权差异。"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT.parent / "csa_ascendc_topk_hc_ep16_20260928/profile_report.py"
    spec = importlib.util.spec_from_file_location("csa_profile_report", source)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    reporter.main(root=ROOT, revision="2d2f9ca0")
    data = json.loads((ROOT / "model/model_gap_rank0.json").read_text())
    weights = {131072: 0.7, 8192: 0.3}
    cases = data["cases"]
    if len(cases) != 2 or {(row["history"], row["batch"]) for row in cases} != {
        (131072, 16), (8192, 16)
    }:
        raise ValueError("必须包含本轮长短B16各一组；不能混入旧档位")
    changes = [{"history": row["history"], "batch": row["batch"],
                "weight": weights[row["history"]],
                "change_pct": (row["pto"]["csa"]["mean_us"] / row["native"]["csa"]["mean_us"] - 1) * 100}
               for row in cases]
    weighted = sum(row["weight"] * row["change_pct"] for row in changes)
    result = {"operator": "2d2f9ca0", "cases": changes, "weighted_csa_change_pct": weighted,
              "scope": "独立三步rank0 profile，同轮Native为基线；完整CSA含首次metadata。"
                       "128K/8K耗时变化率按7:3；不是对旧PTO的单因素收益。"
                       "P95另列，不能由均值权重抵消；最终以正式EP16 forward验收。"}
    (ROOT / "model/weighted_csa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (ROOT / "model/MODEL_GAP.md").open("a") as output:
        output.write(f"\n完整CSA均值变化率按128K/8K七三加权为{weighted:+.3f}%。\n"
                     "这只比较同轮Native；各档P95及模型forward分别验收。"
                     "[原始加权项](weighted_csa.json)。\n")
    print(result)


if __name__ == "__main__":
    main()
