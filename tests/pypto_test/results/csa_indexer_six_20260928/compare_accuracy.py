"""固定规约对照已有CPU状态；不发起设备测试，不放宽逐元素标准。"""

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def main():
    torch.set_num_threads(4)
    common_path = ROOT.parent / "csa_cache_accuracy_20260927/compare.py"
    spec = importlib.util.spec_from_file_location("cache_accuracy", common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    reports, states = {}, {}
    for label in ("baseline", "candidate"):
        folder = ROOT / "accuracy" / label
        reports[label] = json.loads((folder / "report.json").read_text())
        states[label] = torch.load(folder / "states.pt", map_location="cpu", weights_only=True)
    result = {
        "scope": "B16/H32768/S6，长档8192行leaf+causal tail；2a740c1f对照S6，非Native对齐验收",
        "checks": {},
        "errors": [],
    }
    before, after = (reports[k] for k in ("baseline", "candidate"))
    for key in (
        "checkpoint",
        "seed",
        "batch",
        "history",
        "layer_index",
        "weight_nz_mode",
        "variant",
        "deterministic_level",
        "hccl_deterministic",
        "pto_reduction",
        "accuracy_fixture",
    ):
        if before[key] != after[key]:
            result["errors"].append(f"配置不同: {key}")
    for label, report in reports.items():
        for category in ("native_self", "pto_self"):
            if any(v["status"] != "PASS" for v in report[category].values()):
                result["errors"].append(f"固定输入自重放不同: {label}/{category}")
        for category in ("native_guards", "pto_guards"):
            if any(v["status"] != "PASS" for checks in report[category] for v in checks.values()):
                result["errors"].append(f"保护区或metadata失败: {label}/{category}")
        if report["topk_selection"]["structural_errors"]:
            result["errors"].append(f"Top-K结构失败: {label}")
    for side in ("native", "pto"):
        old, new = (states[k][side] for k in ("baseline", "candidate"))
        if set(old) != set(common.NAMES) or set(new) != set(common.NAMES):
            result["errors"].append(f"状态项不全: {side}")
            continue
        result["checks"][side] = {k: compare_tensor(new[k], old[k], 0, 0) for k in common.NAMES}
        if any(v["status"] != "PASS" for v in result["checks"][side].values()):
            result["errors"].append(f"跨版本逐元素不一致: {side}")
    result["graph"] = after.get("graph", after.get("graph_replay", {}))
    if result["graph"].get("status") != "PASS":
        result["errors"].append("候选A/B/A图重放失败")
    result["status"] = "FAIL" if result["errors"] else "PASS"
    (ROOT / "accuracy/comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"], result["errors"], flush=True)
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
