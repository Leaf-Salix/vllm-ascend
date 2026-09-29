"""仅覆盖A常驻改动的ROW32/96尾行，复用短历史和既有padding检查。"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from dsv4_csa_validation import compare_tensor  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def main():
    task = (ROOT / "boundary/task.txt").read_text().strip()
    status = subprocess.check_output(["task-submit", "--status", task], text=True).strip()
    if status != "completed (exit=0)":
        raise RuntimeError(f"等待同一边界任务终态：{status}")
    spec = importlib.util.spec_from_file_location(
        "ob_boundary_common", ROOT.parent / "csa_score_single_root_20260929/collect_boundary.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    torch.set_num_threads(4)
    source = read(ROOT / "source.json")
    result = {"task": task, "task_status": status, "cases": [],
              "scope": "H127/B4(T24→ROW32)与B8(T48→ROW96)，实际改动的两条尾行路径；无性能或EP16结论"}
    for batch in (4, 8):
        folder = ROOT / f"boundary/b{batch}"
        reports, states, contexts = {}, {}, {}
        for side in ("baseline", "candidate"):
            report = reports[side] = read(folder / side / "report.json")
            context = contexts[side] = read(folder / side / "run_context.json")
            if context["source"] != source["source_prefix"] + "-" + side:
                raise ValueError("边界未使用同一冻结私有包")
            if (report["status"], report["batch"], report["history"], report["variant"],
                    report["deterministic_level"], report["effective_weight_nz_mode"],
                    report["pto_reduction"]["atomic_add"]) != (
                    "MEASURED", batch, 127, source["variant"], 1, 2, 0):
                raise ValueError("边界配置不匹配")
            if set(report["pto_self"]) != common.STATES or any(
                    v["status"] != "PASS" for v in report["pto_self"].values()):
                raise ValueError("自身图状态缺失或不一致")
            if any(v["status"] != "PASS" for checks in report["pto_guards"] for v in checks.values()):
                raise ValueError("边界保护区改变")
            if report["topk_selection"]["structural_errors"]:
                raise ValueError("Top-K结构错误")
            padding = report["padding_graph"]
            if padding["status"] != "PASS" or [p["active_batch"] for p in padding["replays"]] != [
                    batch, batch - 1, 1, batch]:
                raise ValueError("固定图padding覆盖不完整")
            states[side] = torch.load(folder / side / "states.pt", map_location="cpu", weights_only=True)["pto"]
            if set(states[side]) != common.STATES:
                raise ValueError("跨版本八类状态缺失")
        if any(reports["baseline"][k] != reports["candidate"][k] for k in (
                "checkpoint", "seed", "layer_index", "accuracy_fixture")):
            raise ValueError("两侧输入配置不同")
        if any(contexts["baseline"][k] != contexts["candidate"][k] for k in ("device", "cann", "variant")):
            raise ValueError("两侧运行环境不同")
        checks = {k: compare_tensor(states["candidate"][k], states["baseline"][k], 0, 0)
                  for k in sorted(common.STATES)}
        result["cases"].append({"batch": batch, "history": 127, "contexts": contexts, "checks": checks,
                                "padding": {s: common.compact_padding(r["padding_graph"], folder / s / "report.json")
                                            for s, r in reports.items()}})
    result["status"] = "PASS" if all(v["status"] == "PASS" for c in result["cases"]
                                   for v in c["checks"].values()) else "FAIL"
    (ROOT / "boundary/summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["status"])
    if result["status"] != "PASS":
        raise SystemExit("边界跨版本状态不一致")


if __name__ == "__main__":
    main()
