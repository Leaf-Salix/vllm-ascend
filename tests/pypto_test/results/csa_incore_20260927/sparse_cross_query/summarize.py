"""Collect existing cross-query pilot results; no device execution."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1]
CASES = ((131072, 4), (131072, 8), (131072, 16),
         (8192, 16), (8192, 24), (8192, 32), (8192, 40))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    common = load_module("csa_matrix_summary", RESULTS / "csa_native_cube_matrix_20260927/summarize.py")
    collector = load_module("early_kv_summary", ROOT.parent / "sparse_kv_early/summarize.py")
    rows = collector.collect(common, "sparse_cross_query", CASES)
    checks = []
    for name, reference in (
        ("standalone", "固定Native Sparse输入；对Native输出的误差，非PTO基底逐bit检查"),
        ("mixed_tail", "B9均匀attention解析值；有效→全无效→有效query及不均分尾部"),
        ("zero_work_cores_b3", "B3均匀attention解析值；T18、6个AIC零工作量"),
    ):
        path = ROOT / name / "report.json"
        if path.exists():
            checks.append({"name": name, "reference": reference, "report": str(path),
                           "comparison": json.loads(path.read_text())["comparison"]})
    result = {
        "base_revision": "c160cabe",
        "patch": "candidate.patch",
        "scope": "同一跨query流水候选；各档独立采集，已有档位不自动代表七档或整模型验收",
        "task_ids": ["task_20260927_165242_22114026823", "task_20260927_165609_22367535890"],
        "fixed_input_vs_baseline": {
            "status": "PASS",
            "elements": 7864320,
            "comparison": "torch.equal(actual, baseline)",
            "baseline": "../sparse_kv_early/gated_standalone/output.pt",
            "actual": "standalone/output.pt",
            "evidence": "task_20260927_165242_22114026823 stdout: CROSS_QUERY_FIXED_INPUT_AND_MIXED_TAIL_BIT_EQUAL_PASS",
        },
        "diagnostics": checks,
        "cases": rows,
    }
    (ROOT / "cases.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for row in rows:
        aic = [w["qk_pv_aic"]["mean_us"] for w in row["swimlane"]]
        merge = [w["merge_norm"]["mean_us"] for w in row["swimlane"]]
        print(row["history"], row["batch"], row["timing"]["body"],
              "AIC", (min(aic), max(aic)) if aic else "pending",
              "merge", (min(merge), max(merge)) if merge else "pending")


if __name__ == "__main__":
    main()
