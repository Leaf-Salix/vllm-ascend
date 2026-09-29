"""复用完整状态/DFX收集，再分开报告HC前段的任务交接与完整CSA。"""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT.parent / "csa_sparse_first_pv_20260929/collect.py"
    spec = importlib.util.spec_from_file_location("hc_priority_collection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.TITLE = "优先完成HC attention前门控，再启动Sinkhorn"
    module.TARGETS = {"pre": "split_pre_post", "comb": "comb_sinkhorn", "mix": "mix_x_rms_norm"}
    module.DESCRIPTION = "只增加pre/post到Sinkhorn的任务依赖；算术、分块、任务数和布局保持。"
    module.main()
    evidence = json.loads((ROOT / "evidence.json").read_text())
    summary = {"task": evidence["task"], "state_status": evidence["state_status"], "cases": []}
    helper = module.load("hc_priority_dependency",
                         module.WORKSPACE / "pypto-lib/.claude/skills/critical-path/scripts/report.py")
    worker = module.load("hc_priority_names", ROOT.parent / "csa_scheduling_20260927/upstream_725/compare.py")
    lines = ["# HC前段任务交接", "", "单位μs，按各窗口首个Worker receive归零；四窗均值。",
             "正式CSA与独立DFX分别报告，任务包络不能解释未同时profile的单个正式样本。", "",
             "| 档位/侧 | pre start/end | comb start/end | mix start/end | 前门控end→mix start |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for case in evidence["cases"]:
        record = {"history": case["history"], "batch": case["batch"], "sides": {}}
        for side in ("baseline", "candidate"):
            windows = case["sides"][side]["windows"]
            fields = {name: {key: [w["tasks"][name][key] for w in windows]
                             for key in ("first_start_us", "last_end_us", "kernel_mean_us")}
                      for name in module.TARGETS.values()}
            pre, comb, mix = (fields[name] for name in module.TARGETS.values())
            gap = [b - a for a, b in zip(pre["last_end_us"], mix["first_start_us"])]
            # Check the intended direct edge on the predetermined window 3.
            p = Path(windows[3]["path"])
            a = helper._build_analysis(p.parent, p.parent, 2)
            names = {worker.canonical(name): tid for tid, name in a.graph.name.items()}
            direct = names["split_pre_post"] in a.preds[names["comb_sinkhorn"]]
            if direct != (side == "candidate"):
                raise ValueError("实际调度图未体现候选的唯一依赖差异")
            record["sides"][side] = {"tasks": fields, "pre_end_to_mix_start": gap,
                                     "pre_to_comb_direct_dependency": direct}
            cells = ["/".join(f"{statistics.mean(v[k]):.3f}" for k in ("first_start_us", "last_end_us"))
                     for v in (pre, comb, mix)]
            lines.append(f"| {case['history']//1024}K/B{case['batch']} {side} | " + " | ".join(cells)
                         + f" | {statistics.mean(gap):.3f} |")
        summary["cases"].append(record)
    (ROOT / "schedule_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines += ["", "仅当完整CSA/P95及状态支持时保留，前段更快不自动等于整体更快。",
              "[正式CSA/状态](RESULTS.md)、[四窗交接及实际依赖](schedule_summary.json)。"]
    (ROOT / "SCHEDULE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
