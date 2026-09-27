"""读取免清零候选的已有单卡样本与状态检查，不启动新测试。"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    for history, batch in ((131072, 4), (8192, 16)):
        folder = ROOT / f"h{history}_b{batch}"
        comparison = json.loads((folder / "comparison.json").read_text())
        if comparison["status"] != "PASS":
            raise ValueError(comparison["errors"])
        row = {"history": history, "batch": batch, "state_check": "PASS", "measurements": {}}
        for side in ("baseline", "candidate"):
            path = folder / side / "report.json"
            report = json.loads(path.read_text())
            row["measurements"][side] = {
                backend: {"samples_us": report["timing"][backend]["samples_us"],
                          "mean_us": statistics.mean(report["timing"][backend]["samples_us"]),
                          "p50_us": report["timing"][backend]["us_p50"],
                          "p95_us": report["timing"][backend]["us_p95"],
                          "max_us": report["timing"][backend]["us_max"]}
                for backend in ("native", "pto")
            }
            row["measurements"][side]["source"] = str(path)
        before, after = (row["measurements"][side] for side in ("baseline", "candidate"))
        row["change_pct"] = {
            backend: (after[backend]["mean_us"] / before[backend]["mean_us"] - 1) * 100
            for backend in ("native", "pto")
        }
        rows.append(row)
        print(history, batch, row["change_pct"])
    result = {"operator": "71153bb3 + candidate.patch", "task": "task_20260928_053518_14737829545",
              "scope": "单卡layer4真实权重、可变scale人工历史；atomic0/det1/mode2；每侧20次图计时",
              "limits": "Native控制同时变快；部分运行期间device1另采七档DFX。不能将全部下降量归因于免清零。"
                        "8类状态精确一致，不包含idx_topk_scores；不是Native逐元素对齐或EP16验收。",
              "cases": rows}
    (ROOT / "single_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
