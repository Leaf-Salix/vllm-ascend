"""只比较保存的CPU张量，不启动设备；cache改造属于数值中性变更。"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402 -- 独立结果目录先加入共用比较器路径

NAMES = ("x_out", "idx_topk", "swa.0", "compressed.0", "state.0",
         "indexer.0", "indexer.1", "indexer_state.0")


def main():
    history = int(sys.argv[1])
    folder = ROOT / "accuracy" / f"h{history}_b4"
    reports, states = {}, {}
    for revision in ("original", "split"):
        reports[revision] = json.loads((folder / revision / "report.json").read_text())
        states[revision] = torch.load(folder / revision / "states.pt", map_location="cpu", weights_only=True)
    result = {
        "history": history, "batch": 4, "before": "original", "after": "split",
        "scope": "第二CSA正式权重与相同seed合成历史，固定规约；检查cache改造前后，非Native算术对齐或token验收",
        "checks": {}, "errors": [],
    }
    old, new = (reports[r] for r in ("original", "split"))
    for field in ("checkpoint", "seed", "batch", "history", "layer_index", "weight_nz_mode",
                  "variant", "deterministic_level", "hccl_deterministic", "pto_reduction", "accuracy_fixture"):
        if old[field] != new[field]:
            result["errors"].append(f"配置不同: {field}")
    for revision, report in reports.items():
        for group in ("native_self", "pto_self"):
            if any(c["status"] != "PASS" for c in report[group].values()):
                result["errors"].append(f"重复执行不一致: {revision}/{group}")
        for group in ("native_guards", "pto_guards"):
            if any(c["status"] != "PASS" for checks in report[group] for c in checks.values()):
                result["errors"].append(f"保护区或metadata失败: {revision}/{group}")
        if report["topk_selection"]["structural_errors"]:
            result["errors"].append(f"Top-K结构失败: {revision}")
    for side in ("native", "pto"):
        before, after = (states[r][side] for r in ("original", "split"))
        if set(before) != set(NAMES) or set(after) != set(NAMES):
            result["errors"].append(f"状态项缺失或未声明: {side}")
            continue
        result["checks"][side] = {name: compare_tensor(after[name], before[name], 0, 0) for name in NAMES}
        if any(c["status"] != "PASS" for c in result["checks"][side].values()):
            result["errors"].append(f"改造前后逐元素不一致: {side}")
    result["native_control_equal"] = bool(result["checks"].get("native")) and all(
        c["status"] == "PASS" for c in result["checks"]["native"].values())
    result["accuracy_fixture"] = new["accuracy_fixture"]
    result["after_vs_native"] = new["pto_native"]
    result["before_vs_native"] = old["pto_native"]
    result["topk_before_after"] = {r: reports[r]["topk_selection"] for r in reports}
    result["graph"] = new.get("graph", new.get("graph_replay", {}))
    if result["graph"].get("status") != "PASS":
        result["errors"].append("新版本A/B/A图重放未通过")
    result["status"] = "FAIL" if result["errors"] else "PASS"
    (folder / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(history, result["status"], result["errors"], flush=True)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
